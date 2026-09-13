use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n_dogs: usize, n_trials: usize, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_dogs = count(&v, "n_dogs")?;
    let n_trials = count(&v, "n_trials")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != n_dogs { return Err("y row count mismatch".into()); }
    let mut y = Vec::with_capacity(n_dogs.checked_mul(n_trials).ok_or("data dimensions overflow")?);
    for row in rows {
        let values = row.as_array().ok_or("y rows must be arrays")?;
        if values.len() != n_trials { return Err("y column count mismatch".into()); }
        for x in values {
            match x.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("y values must be integer 0 or 1".into()) }
        }
    }
    Ok(Data { n_dogs, n_trials, y })
}
fn expected_layout(_d: &Data) -> String { "beta.1\nbeta.2\nbeta.3".into() }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let (b0, b1, b2) = (q[0], q[1], q[2]);
    // beta ~ normal(0, 100): propto=true removes the data-independent normalizer.
    let mut lp = -0.5 * (b0*b0 + b1*b1 + b2*b2) / 10000.0;
    g[0] = -b0 / 10000.0; g[1] = -b1 / 10000.0; g[2] = -b2 / 10000.0;
    // This reproduces the Stan transformed-parameter recurrence for each dog,
    // fused with its Bernoulli-logit contribution without bind-time summaries.
    for dog in 0..d.n_dogs {
        let mut n_avoid = 0.0;
        let mut n_shock = 0.0;
        for trial in 0..d.n_trials {
            let y = d.y[dog * d.n_trials + trial] as f64;
            let eta = b0 + b1 * n_avoid + b2 * n_shock;
            lp += y * eta - softplus(eta);
            let r = y - sigmoid(eta);
            g[0] += r; g[1] += r * n_avoid; g[2] += r * n_shock;
            // Stan sets the next row from the preceding outcome.
            n_avoid += 1.0 - y;
            n_shock += y;
        }
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } }
}
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
    if out.is_null() { return fatal(err,cap,"null bound output"); } unsafe { *out=std::ptr::null_mut(); }
    let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{
        let data=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;
        let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;
        if ndim != 3 || got != expected_layout(&data) { return Err("dimension/layout mismatch".into()); }
        Ok(Bound{data,ndim}) }));
    match answer { Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void) { let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}})); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 { let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")} }
