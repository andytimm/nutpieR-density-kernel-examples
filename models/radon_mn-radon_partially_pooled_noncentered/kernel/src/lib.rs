use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county_idx: Vec<usize>, log_radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x = finite_num(v, key)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 { Err(format!("{key} must be a nonnegative integer")) } else { Ok(x as usize) }
}
fn num_vec(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let vv = Value::Object(o.clone());
    let n = count(&vv, "N")?;
    let j = count(&vv, "J")?;
    let raw_idx = num_vec(&vv, "county_idx", n)?;
    let mut county_idx = Vec::with_capacity(n);
    for x in raw_idx { if x < 1.0 || x.fract() != 0.0 || x > j as f64 { return Err("county_idx outside 1:J".into()); } county_idx.push(x as usize - 1); }
    let log_radon = num_vec(&vv, "log_radon", n)?;
    Ok(Data { n, j, county_idx, log_radon })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.j).map(|i| format!("alpha_raw.{i}")).collect();
    names.extend(["mu_alpha".to_string(), "sigma_alpha".to_string(), "sigma_y".to_string()]);
    names.join("\n")
}
// Exact Stan target: propto=true,jacobian=true. The explicit normal_lpdf
// likelihood retains its normalizing terms; distribution statements do not.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let mu_ix = d.j;
    let sa_ix = d.j + 1;
    let sy_ix = d.j + 2;
    let mu = q[mu_ix];
    let sigma_alpha = q[sa_ix].exp();
    let sigma_y = q[sy_ix].exp();
    if !sigma_alpha.is_finite() || !sigma_y.is_finite() { return Err("scale overflow".into()); }
    g.fill(0.0);
    let mut lp = q[sa_ix] + q[sy_ix]; // Jacobians for two lower=0 transforms
    // alpha_raw ~ std_normal(); mu_alpha ~ normal(0, 10); sigma_* ~ normal(0,1)
    for i in 0..d.j { lp -= 0.5 * q[i] * q[i]; g[i] -= q[i]; }
    lp -= 0.5 * (mu / 10.0) * (mu / 10.0); g[mu_ix] -= mu / 100.0;
    lp -= 0.5 * sigma_alpha * sigma_alpha; g[sa_ix] += 1.0 - sigma_alpha * sigma_alpha;
    lp -= 0.5 * sigma_y * sigma_y; g[sy_ix] += 1.0 - sigma_y * sigma_y;
    let log_sigma_y = q[sy_ix];
    for n in 0..d.n {
        let i = d.county_idx[n];
        let alpha = mu + sigma_alpha * q[i];
        let resid = d.log_radon[n] - alpha;
        let z = resid / sigma_y;
        lp += -0.5 * z * z - log_sigma_y - 0.5 * std::f64::consts::TAU.ln();
        let da = resid / (sigma_y * sigma_y);
        g[i] += da * sigma_alpha;
        g[mu_ix] += da;
        g[sa_ix] += da * sigma_alpha * q[i];
        g[sy_ix] += z * z - 1.0;
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
 if out.is_null() { return fatal(err,cap,"null bound output"); } unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != d.j+3 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
