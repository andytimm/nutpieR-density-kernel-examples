use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

const I: usize = 21;
struct Data { n: [f64; I], n_total: [f64; I], x1: [f64; I], x2: [f64; I] }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn fixed_array(v: &Value, key: &str) -> Result<[f64; I], String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != I { return Err(format!("{key} must have length {I}")); }
    let mut out = [0.0; I];
    for (i, x) in a.iter().enumerate() { out[i] = number(x, key)?; }
    Ok(out)
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let iv = o.get("I").ok_or("I is required")?;
    let i = number(iv, "I")?;
    if i != I as f64 { return Err("I must equal 21".into()); }
    let n = fixed_array(&v, "n")?;
    let n_total = fixed_array(&v, "N")?;
    let x1 = fixed_array(&v, "x1")?;
    let x2 = fixed_array(&v, "x2")?;
    for j in 0..I {
        if n[j] < 0.0 || n_total[j] < 0.0 || n[j] > n_total[j] || n[j].fract() != 0.0 || n_total[j].fract() != 0.0 { return Err("n and N must be nonnegative integers with n <= N".into()); }
    }
    Ok(Data { n, n_total, x1, x2 })
}
fn expected_layout() -> String {
    let mut v = vec!["alpha0".to_string(), "alpha1".to_string(), "alpha12".to_string(), "alpha2".to_string()];
    for j in 1..=I { v.push(format!("b.{j}")); }
    v.push("sigma".to_string()); v.join("\n")
}
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let z=x.exp(); z/(1.0+z) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let (a0, a1, a12, a2) = (q[0], q[1], q[2], q[3]);
    let log_sigma = q[25]; let sigma = log_sigma.exp();
    if !sigma.is_finite() { return Err("sigma transform is non-finite".into()); }
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let mut lp = -0.5 * (a0*a0 + a1*a1 + a12*a12 + a2*a2) - (sigma*sigma).ln_1p() + log_sigma;
    g.fill(0.0);
    g[0] = -a0; g[1] = -a1; g[2] = -a12; g[3] = -a2;
    let mut gs = 1.0 - 2.0 * sigma*sigma / (1.0 + sigma*sigma);
    for j in 0..I {
        let b = q[4+j];
        lp += -0.5*b*b*inv_sigma2 - log_sigma;
        let eta = a0 + a1*d.x1[j] + a2*d.x2[j] + a12*d.x1[j]*d.x2[j] + b;
        lp += d.n[j]*eta - d.n_total[j]*softplus(eta);
        let r = d.n[j] - d.n_total[j]*sigmoid(eta);
        g[0] += r; g[1] += r*d.x1[j]; g[2] += r*d.x1[j]*d.x2[j]; g[3] += r*d.x2[j];
        g[4+j] = r - b*inv_sigma2;
        gs += b*b*inv_sigma2 - 1.0;
    }
    g[25] = gs;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=26||got!=expected_layout(){return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
