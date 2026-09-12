use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

const SCALE: f64 = 2.5;
struct Data { y: Vec<f64>, x: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn numeric_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>()
        .filter(|a| a.iter().all(|v| v.is_finite()))
        .ok_or_else(|| format!("{key} must be finite numeric array"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    let y = numeric_array(o, "kid_score", n)?;
    let x = numeric_array(o, "mom_hs", n)?;
    if y.iter().any(|&z| !(0.0..=200.0).contains(&z)) { return Err("kid_score outside [0, 200]".into()); }
    if x.iter().any(|&z| !(0.0..=1.0).contains(&z)) { return Err("mom_hs outside [0, 1]".into()); }
    Ok(Data { y, x })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|v| !v.is_finite()) { return Err("non-finite position".into()); }
    let sigma = q[2].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform overflow/underflow".into()); }
    let inv_sigma2 = 1.0 / (sigma * sigma);
    let mut sum_r = 0.0;
    let mut sum_rx = 0.0;
    let mut sum_r2 = 0.0;
    // Fused original per-observation normal likelihood and reverse gradient.
    for (&y, &x) in d.y.iter().zip(&d.x) {
        let r = y - q[0] - q[1] * x;
        sum_r += r;
        sum_rx += r * x;
        sum_r2 += r * r;
    }
    let n = d.y.len() as f64;
    let scaled_ss = sum_r2 * inv_sigma2;
    let sigma2 = sigma * sigma;
    let cauchy_term = -((sigma / SCALE) * (sigma / SCALE)).ln_1p();
    g[0] = sum_r * inv_sigma2;
    g[1] = sum_rx * inv_sigma2;
    // Includes lower-bound transform Jacobian after all constrained-density terms.
    g[2] = -n + scaled_ss - 2.0 * sigma2 / (SCALE * SCALE + sigma2) + 1.0;
    Ok(-n * sigma.ln() - 0.5 * scaled_ss + cauchy_term + q[2])
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 {
        let n = s.len().min(cap - 1);
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; }
    }
}
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); }; unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?; let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != 3 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); }; unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) }));
    match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in workspace") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 {
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); }
        let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); }
        let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }; unsafe { *lp = value; }; Ok(0)
    }));
    match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in evaluate") }
}
