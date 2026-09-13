use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// The centered predictors and interaction are exactly the Stan transformed-data block.
struct Data { y: Vec<f64>, hs: Vec<f64>, iq: Vec<f64>, inter: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be nonnegative integer")? as usize;
    if n == 0 { return Err("N must be positive for Stan mean() transformed data".into()); }
    let y = array(&v, "kid_score", n)?;
    let mut hs = array(&v, "mom_hs", n)?;
    let mut iq = array(&v, "mom_iq", n)?;
    // This is the original transformed-data work, not a likelihood reduction.
    let mean_hs = hs.iter().sum::<f64>() / n as f64;
    let mean_iq = iq.iter().sum::<f64>() / n as f64;
    for x in &mut hs { *x -= mean_hs; }
    for x in &mut iq { *x -= mean_iq; }
    let inter = hs.iter().zip(&iq).map(|(a,b)| a*b).collect();
    Ok(Data { y, hs, iq, inter })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let sigma = q[4].exp();
    let inv_var = 1.0 / (sigma * sigma);
    let mut lp = -(d.y.len() as f64 - 1.0) * q[4]; // -N log(sigma) plus lower-bound Jacobian
    g.fill(0.0);
    let mut ss = 0.0;
    for i in 0..d.y.len() {
        let eta = q[0] + q[1]*d.hs[i] + q[2]*d.iq[i] + q[3]*d.inter[i];
        let r = d.y[i] - eta;
        lp -= 0.5 * r * r * inv_var;
        let w = r * inv_var;
        g[0] += w; g[1] += w*d.hs[i]; g[2] += w*d.iq[i]; g[3] += w*d.inter[i];
        ss += r*r*inv_var;
    }
    g[4] = ss - d.y.len() as f64 + 1.0;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char, cap:usize, s:impl AsRef<str>)->i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=5||got!=expected_layout(&d){return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gp:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||gp.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gp,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|z|!z.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
