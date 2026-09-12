use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Input values are copied at bind.  The two likelihood vectors are kept in
// original observation order; no data-only reductions are formed.
struct Data { n: usize, encouraged: Vec<f64>, watched: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn nonnegative_count(v: &Value, key: &str) -> Result<usize, String> {
    let x = finite_num(v, key)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 { Err(format!("{key} must be a nonnegative integer")) }
    else { Ok(x as usize) }
}
fn finite_vector(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length does not match N")); }
    a.iter().enumerate().map(|(i, x)| x.as_f64().filter(|z| z.is_finite())
        .ok_or_else(|| format!("{key}[{}] must be finite numeric", i + 1))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let obj = Value::Object(o.clone());
    let n = nonnegative_count(&obj, "N")?;
    let encouraged = finite_vector(&obj, "encouraged", n)?;
    let watched = finite_vector(&obj, "watched", n)?;
    Ok(Data { n, encouraged, watched })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nsigma".into() }

// Stan: watched ~ normal(beta[1] + beta[2] * encouraged, sigma).
// q[2] is log(sigma); BridgeStan's jacobian=true adds q[2].  With
// propto=true the Normal constant is omitted, while -N*log(sigma) remains.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let sigma = q[2].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("scale transform is non-finite".into()); }
    let inv_sigma = 1.0 / sigma;
    let inv_var = inv_sigma * inv_sigma;
    let mut sum_sq = 0.0;
    let mut grad_intercept = 0.0;
    let mut grad_slope = 0.0;
    for i in 0..d.n {
        let residual = d.watched[i] - (q[0] + q[1] * d.encouraged[i]);
        sum_sq += residual * residual;
        let scaled = residual * inv_var;
        grad_intercept += scaled;
        grad_slope += d.encouraged[i] * scaled;
    }
    g[0] = grad_intercept;
    g[1] = grad_slope;
    g[2] = -(d.n as f64) + sum_sq * inv_var + 1.0;
    Ok(-(d.n as f64) * q[2] - 0.5 * sum_sq * inv_var + q[2])
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
    if out.is_null() { return fatal(err, cap, "null bound output"); }; unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> { let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?; let data = parse_data(v)?; let want = expected_layout(&data); let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?; if ndim != 3 || got != want { return Err("dimension/layout mismatch".into()); } Ok(Bound { data, ndim }) }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 { if out.is_null() { return fatal(err, cap, "null workspace output"); }; unsafe { *out = std::ptr::null_mut(); }; let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) })); match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace") } }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 { let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> { if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); } let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); } let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) }; let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0) })); match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in evaluate") } }
