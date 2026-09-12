use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { floor_measure: Vec<f64>, log_radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn finite_vector(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>()
        .filter(|a| a.iter().all(|x| x.is_finite()))
        .ok_or_else(|| format!("{key} must contain finite numerics"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let input = Value::Object(o.clone());
    let n = finite_num(&input, "N")?;
    if n < 0.0 || n.fract() != 0.0 || n > usize::MAX as f64 { return Err("N must be a nonnegative integer".into()); }
    let n = n as usize;
    Ok(Data { floor_measure: finite_vector(&input, "floor_measure", n)?, log_radon: finite_vector(&input, "log_radon", n)? })
}
fn expected_layout(_: &Data) -> String { "alpha\nbeta\nsigma_y".into() }

// Matches propto=true, jacobian=true. The tilde priors use lupdf terms; the
// explicit normal_lpdf likelihood retains its -0.5*log(2*pi) per observation.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let alpha = q[0];
    let beta = q[1];
    let sigma = q[2].exp();
    if !sigma.is_finite() { return Err("sigma transform overflow".into()); }
    let inv_sigma = 1.0 / sigma;
    let inv_sigma_sq = inv_sigma * inv_sigma;
    let mut lp = -0.5 * (alpha / 10.0).powi(2) - 0.5 * (beta / 10.0).powi(2)
        - 0.5 * sigma * sigma + q[2];
    let mut ga = -alpha / 100.0;
    let mut gb = -beta / 100.0;
    let mut gs_constrained = -sigma;
    for (&x, &y) in d.floor_measure.iter().zip(&d.log_radon) {
        let residual = y - (alpha + beta * x);
        let z = residual * inv_sigma;
        lp += -0.5 * z * z - q[2] - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let scaled = residual * inv_sigma_sq;
        ga += scaled;
        gb += x * scaled;
        gs_constrained += residual * residual * inv_sigma_sq * inv_sigma - inv_sigma;
    }
    g[0] = ga;
    g[1] = gb;
    g[2] = gs_constrained * sigma + 1.0;
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
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> { let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?; let data = parse_data(v)?; let want = expected_layout(&data); let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?; if ndim != 3 || got != want { return Err("dimension/layout mismatch".into()); } Ok(Bound { data, ndim }) }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 { if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); } let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) })); match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace") } }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 { let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> { if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); } let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); } let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) }; let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0) })); match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in evaluate") } }
