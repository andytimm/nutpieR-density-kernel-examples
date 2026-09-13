use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { v: usize, m: usize, n: usize, w: Vec<usize>, doc: Vec<usize> }
struct Bound { data: Data, ndim: usize }
struct Workspace { theta: Vec<f64>, phi: Vec<f64>, dtheta: Vec<f64>, dphi: Vec<f64>, qphi: Vec<f64> }

fn pos_int(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a positive integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn index_array(v: &Value, key: &str, n: usize, upper: usize) -> Result<Vec<usize>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|x| {
        let z=x.as_u64().ok_or_else(|| format!("{key} must contain integers"))?;
        let z=usize::try_from(z).map_err(|_| format!("{key} value too large"))?;
        if z == 0 || z > upper { Err(format!("{key} value out of bounds")) } else { Ok(z-1) }
    }).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let vv=pos_int(&Value::Object(o.clone()), "V")?;
    let m=pos_int(&Value::Object(o.clone()), "M")?;
    let n=pos_int(&Value::Object(o.clone()), "N")?;
    if vv < 2 { return Err("V must be at least 2".into()); }
    let w=index_array(&Value::Object(o.clone()), "w", n, vv)?;
    let doc=index_array(&Value::Object(o.clone()), "doc", n, m)?;
    Ok(Data { v: vv, m, n, w, doc })
}
fn expected_layout(d: &Data) -> String {
    let mut out=Vec::with_capacity(d.m + 2*(d.v-1));
    for m in 1..=d.m { out.push(format!("theta.{m}.1")); }
    // BridgeStan's last-index-major layout interleaves the two array elements.
    for j in 1..d.v { for k in 1..=2 { out.push(format!("phi.{k}.{j}")); } }
    out.join("\n")
}
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0/(1.0+(-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn simplex_v(q: &[f64], x: &mut [f64]) -> f64 {
    // Stan 2.8 uses the ILR inverse, not the older stick-breaking transform.
    let k=x.len(); let n=k-1; let mut logits=vec![0.0; k]; let mut sum_w=0.0;
    for i in (1..=n).rev() { let w=q[i-1]/((i*(i+1)) as f64).sqrt(); sum_w += w; logits[i-1] += sum_w; logits[i] -= w*(i as f64); }
    let max=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max); let denom: f64=logits.iter().map(|z|(z-max).exp()).sum();
    for i in 0..k { x[i]=(logits[i]-max).exp()/denom; }
    x.iter().map(|z|z.ln()).sum::<f64>() + 0.5*(k as f64).ln()
}
fn simplex_grad(x: &[f64], a: &[f64], out: &mut [f64]) {
    let k=x.len(); let mean: f64=x.iter().zip(a).map(|(x,a)|x*a).sum();
    let mut prefix=0.0; let mut h=vec![0.0;k];
    for i in 0..k { h[i]=x[i]*(a[i]-mean) + 1.0-(k as f64)*x[i]; }
    for j in 0..k-1 { prefix += h[j]; let r=j+1; out[j]=(prefix-(r as f64)*h[r])/((r*(r+1)) as f64).sqrt(); }
}
fn eval_model(d: &Data, w: &mut Workspace, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    w.dtheta.fill(0.0); w.dphi.fill(0.0); g.fill(0.0);
    let mut lp=0.0;
    for m in 0..d.m { lp += simplex_v(&q[m..m+1], &mut w.theta[2*m..2*m+2]); }
    for k in 0..2 {
        // Raw names are dot-indexed/interleaved, but the Stan transform consumes
        // each array element's simplex coordinates contiguously (verified by probe).
        for j in 0..d.v-1 { w.qphi[j] = q[d.m + k*(d.v-1) + j]; }
        lp += simplex_v(&w.qphi, &mut w.phi[k*d.v..(k+1)*d.v]);
    }
    for n in 0..d.n {
        let m=d.doc[n]; let v=d.w[n];
        let t0=w.theta[2*m]; let t1=w.theta[2*m+1];
        let p0=w.phi[v]; let p1=w.phi[d.v+v]; let mix=t0*p0+t1*p1;
        if !mix.is_finite() || mix <= 0.0 { return Err("nonpositive mixture probability".into()); }
        lp += mix.ln();
        w.dtheta[2*m] += p0/mix; w.dtheta[2*m+1] += p1/mix;
        w.dphi[v] += t0/mix; w.dphi[d.v+v] += t1/mix;
    }
    for m in 0..d.m { simplex_grad(&w.theta[2*m..2*m+2], &w.dtheta[2*m..2*m+2], &mut g[m..m+1]); }
    for k in 0..2 { simplex_grad(&w.phi[k*d.v..(k+1)*d.v], &w.dphi[k*d.v..(k+1)*d.v], &mut w.qphi); for i in 0..d.v-1 { g[d.m+k*(d.v-1)+i]=w.qphi[i]; } }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())} else {Ok(unsafe{slice::from_raw_parts(p.cast(),n)})} }
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let want_ndim=data.m+2*(data.v-1);if ndim!=want_ndim||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};let b=unsafe{&*bound.cast::<Bound>()};let w=Workspace{theta:vec![0.0;2*b.data.m],phi:vec![0.0;2*b.data.v],dtheta:vec![0.0;2*b.data.m],dphi:vec![0.0;2*b.data.v],qphi:vec![0.0;b.data.v-1]};Ok(Box::into_raw(Box::new(w)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let w=unsafe{&mut *workspace.cast::<Workspace>()};let value=match eval_model(&b.data,w,q,g){Ok(x)=>x,Err(_)=>return Ok(1)};if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
