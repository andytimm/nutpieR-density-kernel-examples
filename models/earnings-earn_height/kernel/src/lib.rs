use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { earn: Vec<f64>, height: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn positive_integer(v: &Value, key: &str) -> Result<usize, String> {
    let n = v.get(key).and_then(Value::as_f64)
        .filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0)
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    if n > usize::MAX as f64 { return Err(format!("{key} is too large")); }
    Ok(n as usize)
}
fn finite_vector(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite())
        .ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let n = positive_integer(&v, "N")?;
    let earn = finite_vector(&v, "earn", n)?;
    let height = finite_vector(&v, "height", n)?;
    Ok(Data { earn, height })
}
fn expected_layout() -> &'static str { "beta.1\nbeta.2\nsigma" }

// Stan normal likelihood with propto=true and sigma = exp(q[2]).  The +q[2]
// is the Jacobian of the lower-bound transform.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let log_sigma = q[2];
    let sigma = log_sigma.exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform is not finite".into()); }
    let inv_var = 1.0 / (sigma * sigma);
    if !inv_var.is_finite() { return Err("sigma inverse variance is not finite".into()); }
    let mut lp = log_sigma; // Jacobian
    let mut gb0 = 0.0;
    let mut gb1 = 0.0;
    let mut gs = 1.0; // Jacobian derivative
    for (&y, &h) in d.earn.iter().zip(&d.height) {
        let residual = y - (q[0] + q[1] * h);
        let scaled_sq = residual * residual * inv_var;
        lp += -log_sigma - 0.5 * scaled_sq;
        let d_eta = residual * inv_var;
        gb0 += d_eta;
        gb1 += d_eta * h;
        gs += -1.0 + scaled_sq;
    }
    g[0] = gb0; g[1] = gb1; g[2] = gs;
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

#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let value: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(value)?;
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != 3 || got != expected_layout() { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match result { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) }));
    match result { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in workspace") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); }
        let b = unsafe { &*bound.cast::<Bound>() };
        if ndim != b.ndim || q.is_null() { return Err("evaluation dimension".into()); }
        let q = unsafe { slice::from_raw_parts(q, ndim) }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = eval_model(&b.data, q, g)?;
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }
        unsafe { *lp = value; }; Ok(0)
    }));
    match result { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in evaluate") }
}
