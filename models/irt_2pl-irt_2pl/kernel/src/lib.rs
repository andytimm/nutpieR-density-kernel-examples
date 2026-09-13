use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { i: usize, j: usize, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn int_field(o: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    o.get(key).and_then(Value::as_u64).and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let i = int_field(o, "I")?;
    let j = int_field(o, "J")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != i { return Err("y has wrong row count".into()); }
    let mut y = Vec::with_capacity(i.checked_mul(j).ok_or("data dimensions overflow")?);
    for row in rows {
        let a = row.as_array().ok_or("y must be a two-dimensional array")?;
        if a.len() != j { return Err("y has wrong column count".into()); }
        for x in a {
            match x.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("y entries must be 0 or 1".into()) }
        }
    }
    Ok(Data { i, j, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(4 + d.i * 2 + d.j);
    names.push("sigma_theta".to_owned());
    for k in 1..=d.j { names.push(format!("theta.{k}")); }
    names.push("sigma_a".to_owned());
    for k in 1..=d.i { names.push(format!("a.{k}")); }
    names.push("mu_b".to_owned()); names.push("sigma_b".to_owned());
    for k in 1..=d.i { names.push(format!("b.{k}")); }
    names.join("\n")
}
#[inline] fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
#[inline] fn cauchy_positive_u(u: f64) -> (f64, f64) {
    let s = u.exp(); let t2 = (s * 0.5).powi(2);
    (-t2.ln_1p() + u, 1.0 - 2.0 * t2 / (1.0 + t2))
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let st_u= q[0]; let st=st_u.exp();
    let theta=&q[1..1+d.j];
    let sa_idx=1+d.j; let sa_u=q[sa_idx]; let sa=sa_u.exp();
    let a_u=&q[sa_idx+1..sa_idx+1+d.i];
    let mu_idx=sa_idx+1+d.i; let mu=q[mu_idx];
    let sb_idx=mu_idx+1; let sb_u=q[sb_idx]; let sb=sb_u.exp();
    let b=&q[sb_idx+1..sb_idx+1+d.i];
    let st2=st*st; let sa2=sa*sa; let sb2=sb*sb;
    let (mut lp, gst)=cauchy_positive_u(st_u); g[0]=gst;
    let mut sum_theta_sq=0.0;
    for k in 0..d.j { let x=theta[k]; sum_theta_sq += x*x; lp += -0.5*x*x/st2 - st_u; g[1+k] = -x/st2; }
    g[0] += sum_theta_sq/st2 - d.j as f64;
    let (lpa, gsa0)=cauchy_positive_u(sa_u); lp += lpa; g[sa_idx]=gsa0;
    let mut sum_au_sq=0.0;
    for k in 0..d.i { let u=a_u[k]; sum_au_sq += u*u; lp += -0.5*u*u/sa2 - sa_u - 0.5 * std::f64::consts::TAU.ln(); g[sa_idx+1+k] = -u/sa2; }
    g[sa_idx] += sum_au_sq/sa2 - d.i as f64;
    lp += -0.5 * mu * mu / 25.0; g[mu_idx] = -mu / 25.0;
    let (lpb, gsb0)=cauchy_positive_u(sb_u); lp += lpb; g[sb_idx]=gsb0;
    let mut sum_bres_sq=0.0;
    for k in 0..d.i { let r=b[k]-mu; sum_bres_sq += r*r; lp += -0.5*r*r/sb2 - sb_u; g[sb_idx+1+k] = -r/sb2; g[mu_idx] += r/sb2; }
    g[sb_idx] += sum_bres_sq/sb2 - d.i as f64;
    for ii in 0..d.i {
        let a=a_u[ii].exp(); let bi=b[ii]; let mut gau=0.0; let mut gbi=0.0;
        for jj in 0..d.j {
            let eta=a*(theta[jj]-bi); let yy=d.y[ii*d.j+jj] as f64;
            lp += yy*eta-softplus(eta);
            let de=yy-sigmoid(eta);
            g[1+jj] += a*de;
            gau += a*(theta[jj]-bi)*de;
            gbi -= a*de;
        }
        g[sa_idx+1+ii] += gau;
        g[sb_idx+1+ii] += gbi;
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())} else {Ok(unsafe{slice::from_raw_parts(p.cast(),n)})} }
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){ if !p.is_null()&&cap!=0 {let n=s.len().min(cap-1); unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}} }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 {unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=1+data.j+1+data.i+1+1+data.i||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
