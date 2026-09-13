use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, kid_score: Vec<f64>, mom_hs: Vec<f64>, mom_iq: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_f = o.get("N").and_then(Value::as_f64).filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0)
        .ok_or("N must be a nonnegative integer")?;
    if n_f > usize::MAX as f64 { return Err("N is too large".into()); }
    let n = n_f as usize;
    Ok(Data { n, kid_score: finite_array(o, "kid_score", n)?, mom_hs: finite_array(o, "mom_hs", n)?, mom_iq: finite_array(o, "mom_iq", n)? })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let sigma = q[4].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform is non-finite".into()); }
    let inv_var = 1.0 / (sigma * sigma);
    if !inv_var.is_finite() { return Err("sigma transform is out of range".into()); }
    g.fill(0.0);
    let mut sum_sq = 0.0;
    for i in 0..d.n {
        // These are exactly the original Stan transformed-data expressions, kept per evaluation.
        let c2_hs = d.mom_hs[i] - 0.5;
        let c2_iq = d.mom_iq[i] - 100.0;
        let inter = c2_hs * c2_iq;
        let r = d.kid_score[i] - (q[0] + q[1] * c2_hs + q[2] * c2_iq + q[3] * inter);
        let scaled_r = r * inv_var;
        sum_sq += r * r;
        g[0] += scaled_r;
        g[1] += scaled_r * c2_hs;
        g[2] += scaled_r * c2_iq;
        g[3] += scaled_r * inter;
    }
    // normal_lpdf with propto=true retains -log(sigma); sigma=exp(q[4]) adds its Jacobian q[4].
    let u = q[4];
    g[4] = sum_sq * inv_var - d.n as f64 + 1.0;
    Ok(-0.5 * sum_sq * inv_var - d.n as f64 * u + u)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())};2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != 5 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
