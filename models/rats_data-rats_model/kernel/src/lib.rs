use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, npts: usize, rat: Vec<usize>, x: Vec<f64>, y: Vec<f64>, xbar: f64 }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn positive_int(v: &Value, key: &str) -> Result<usize, String> {
    let x = finite_num(v, key)?;
    if x < 1.0 || x.fract() != 0.0 || x > usize::MAX as f64 { return Err(format!("{key} must be a positive integer")); }
    Ok(x as usize)
}
fn num_array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=v.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().enumerate().map(|(i,x)| x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{key}[{i}] must be finite"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    if !v.is_object() { return Err("data must be a JSON object".into()); }
    let n=positive_int(&v,"N")?; let npts=positive_int(&v,"Npts")?;
    let x=num_array(&v,"x",npts)?; let y=num_array(&v,"y",npts)?; let xbar=finite_num(&v,"xbar")?;
    let a=v.get("rat").and_then(Value::as_array).ok_or("rat must be an array")?;
    if a.len()!=npts { return Err("rat length mismatch".into()); }
    let mut rat=Vec::with_capacity(npts);
    for (i,z) in a.iter().enumerate() { let r=z.as_f64().ok_or_else(||format!("rat[{i}] must be numeric"))?;
        if !r.is_finite() || r.fract()!=0.0 || r<1.0 || r>n as f64 { return Err(format!("rat[{i}] out of bounds")); } rat.push(r as usize-1); }
    Ok(Data{n,npts,rat,x,y,xbar})
}
fn expected_layout(d: &Data) -> String {
    let mut names=Vec::with_capacity(2*d.n+5);
    for i in 1..=d.n { names.push(format!("alpha.{i}")); }
    for i in 1..=d.n { names.push(format!("beta.{i}")); }
    names.extend(["mu_alpha".into(),"mu_beta".into(),"sigma_y".into(),"sigma_alpha".into(),"sigma_beta".into()]); names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Option<f64> {
    // Finite positions can have log-scales large enough that `exp(log_scale)`
    // or its square overflows.  Work with inverse scales instead.  This is
    // algebraically the same normal density, but keeps large positive scales
    // finite.  A non-finite intermediate is a recoverable evaluation rejection.
    if q.iter().any(|z| !z.is_finite()) { return None; }
    g.fill(0.0);
    let n = d.n;
    let ma = q[2*n];
    let mb = q[2*n+1];
    let ly = q[2*n+2];
    let la = q[2*n+3];
    let lb = q[2*n+4];
    let iy = (-ly).exp();
    let ia = (-la).exp();
    let ib = (-lb).exp();
    if !iy.is_finite() || !ia.is_finite() || !ib.is_finite() { return None; }
    let mut lp = -0.5*(ma/100.0).powi(2) - 0.5*(mb/100.0).powi(2);
    g[2*n] = -ma/10000.0;
    g[2*n+1] = -mb/10000.0;
    let mut ssy = 0.0;
    let mut ssa = 0.0;
    let mut ssb = 0.0;
    for i in 0..n {
        let da = q[i] - ma;
        let db = q[n+i] - mb;
        if !da.is_finite() || !db.is_finite() { return None; }
        let za = da * ia;
        let zb = db * ib;
        if !za.is_finite() || !zb.is_finite() { return None; }
        lp += -la - 0.5*za*za - lb - 0.5*zb*zb;
        g[i] -= za * ia;
        g[n+i] -= zb * ib;
        g[2*n] += za * ia;
        g[2*n+1] += zb * ib;
        ssa += za*za;
        ssb += zb*zb;
    }
    for j in 0..d.npts {
        let i = d.rat[j];
        let dx = d.x[j] - d.xbar;
        let r = d.y[j] - (q[i] + q[n+i]*dx);
        if !dx.is_finite() || !r.is_finite() { return None; }
        let zy = r * iy;
        if !zy.is_finite() { return None; }
        lp += -ly - 0.5*zy*zy;
        let z = zy * iy;
        if !z.is_finite() { return None; }
        g[i] += z;
        g[n+i] += dx*z;
        ssy += zy*zy;
    }
    // Log-scale transforms: chain rule plus one Jacobian each.
    g[2*n+2] = ssy - d.npts as f64 + 1.0;
    g[2*n+3] = ssa - n as f64 + 1.0;
    g[2*n+4] = ssb - n as f64 + 1.0;
    lp += ly + la + lb;
    if lp.is_finite() && g.iter().all(|x| x.is_finite()) { Some(lp) } else { None }
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a [u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=2*data.n+5||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast();}0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w;}0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into());}let qs=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=match eval_model(&b.data,qs,g){Some(value)=>value,None=>return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
