use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { earn: Vec<f64>, height: Vec<f64>, male: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn vector(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(Value::as_f64).enumerate().map(|(i, x)|
        x.filter(|z| z.is_finite()).ok_or_else(|| format!("{key}[{}] must be finite numeric", i + 1))
    ).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_f = num(&Value::Object(o.clone()), "N")?;
    if n_f < 0.0 || n_f.fract() != 0.0 || n_f > usize::MAX as f64 { return Err("N must be a nonnegative integer".into()); }
    let n = n_f as usize;
    let earn = vector(&Value::Object(o.clone()), "earn", n)?;
    if earn.iter().any(|x| *x <= 0.0) { return Err("earn must be positive for log transform".into()); }
    Ok(Data { earn, height: vector(&Value::Object(o.clone()), "height", n)?, male: vector(&Value::Object(o.clone()), "male", n)? })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let log_sigma = q[4];
    let sigma_inv = (-log_sigma).exp();
    if !sigma_inv.is_finite() { return Err("sigma transform overflow".into()); }
    let inv_var = sigma_inv * sigma_inv;
    if !inv_var.is_finite() { return Err("sigma inverse square overflow".into()); }
    let (b0, b1, b2, b3) = (q[0], q[1], q[2], q[3]);
    let mut lp = log_sigma; // lower=0 sigma transform Jacobian
    let mut gs = 1.0;
    g.fill(0.0);
    // Fused original transformed-data expressions and individual normal terms.
    for i in 0..d.earn.len() {
        let y = d.earn[i].ln();
        let h = d.height[i];
        let m = d.male[i];
        let inter = h * m;
        let resid = y - (b0 + b1 * h + b2 * m + b3 * inter);
        let scaled = resid * sigma_inv;
        lp += -log_sigma - 0.5 * scaled * scaled;
        let r_inv_var = resid * inv_var;
        g[0] += r_inv_var;
        g[1] += r_inv_var * h;
        g[2] += r_inv_var * m;
        g[3] += r_inv_var * inter;
        gs += -1.0 + scaled * scaled;
    }
    g[4] = gs;
    Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=5||got!=expected_layout(&d){return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
