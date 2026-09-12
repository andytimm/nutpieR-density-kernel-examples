use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { log_earn: Vec<f64>, height: Vec<f64>, male: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number(v: &Value, name: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{name} must contain finite numbers"))
}
fn numeric_vector(o: &serde_json::Map<String, Value>, name: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(name).and_then(Value::as_array).ok_or_else(|| format!("{name} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{name} length mismatch")); }
    a.iter().map(|x| number(x, name)).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    let earn = numeric_vector(o, "earn", n)?;
    let height = numeric_vector(o, "height", n)?;
    let male = numeric_vector(o, "male", n)?;
    let mut log_earn = Vec::with_capacity(n);
    for x in earn { if x <= 0.0 { return Err("earn must be > 0".into()); } log_earn.push(x.ln()); }
    Ok(Data { log_earn, height, male })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let sigma = q[3].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("invalid sigma transform".into()); }
    let inv_sigma = 1.0 / sigma;
    let inv_sigma2 = inv_sigma * inv_sigma;
    let (b0, b1, b2) = (q[0], q[1], q[2]);
    let mut lp = 0.0;
    let mut gb0 = 0.0; let mut gb1 = 0.0; let mut gb2 = 0.0; let mut gs = 0.0;
    for i in 0..d.log_earn.len() {
        let residual = d.log_earn[i] - (b0 + b1 * d.height[i] + b2 * d.male[i]);
        let scaled = residual * inv_sigma;
        lp -= 0.5 * scaled * scaled;
        let common = residual * inv_sigma2;
        gb0 += common;
        gb1 += common * d.height[i];
        gb2 += common * d.male[i];
        gs += scaled * scaled;
    }
    // normal_lpdf(... | ..., sigma), propto=true omits only 0.5*log(2*pi).
    // sigma=exp(q[3]); the lower-bound transform contributes +q[3].
    let n = d.log_earn.len() as f64;
    lp -= n * q[3];
    lp += q[3];
    g[0] = gb0; g[1] = gb1; g[2] = gb2; g[3] = gs - n + 1.0;
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
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?; let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != 4 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) }));
    match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 {
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); }
        let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); }
        let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0)
    }));
    match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in evaluate") }
}
