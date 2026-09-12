use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { x: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_array(o: &serde_json::Map<String, Value>, key: &str) -> Result<Vec<f64>, String> {
    o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?
        .iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must contain finite numerics"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    let x = finite_array(o, "x")?;
    let y = finite_array(o, "Y")?;
    if x.len() != n || y.len() != n { return Err("N must match x and Y lengths".into()); }
    if x.iter().any(|z| *z < 0.0) { return Err("x must be nonnegative for real pow".into()); }
    Ok(Data { x, y })
}
fn expected_layout(_: &Data) -> String { "alpha\nbeta\nlambda\ntau".into() }
fn inv_logit(q: f64) -> f64 { if q >= 0.0 { 1.0 / (1.0 + (-q).exp()) } else { let e = q.exp(); e / (1.0 + e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let alpha = q[0];
    let beta = q[1];
    let s = inv_logit(q[2]);
    let lambda = 0.5 + 0.5 * s;
    let tau = q[3].exp();
    if !lambda.is_finite() || !tau.is_finite() { return Err("non-finite transform".into()); }
    let mut lp = -0.5 * alpha * alpha / 1_000_000.0 - 0.5 * beta * beta / 1_000_000.0;
    let mut ga = -alpha / 1_000_000.0;
    let mut gb = -beta / 1_000_000.0;
    let mut glambda = 0.0;
    let mut sum_sq = 0.0;
    for (&x, &y) in d.x.iter().zip(&d.y) {
        let lx = lambda.powf(x);
        let residual = y - alpha + beta * lx;
        lp += -0.5 * tau * residual * residual + 0.5 * q[3];
        ga += tau * residual;
        gb += -tau * residual * lx;
        if x != 0.0 { glambda += -tau * residual * beta * x * lambda.powf(x - 1.0); }
        sum_sq += residual * residual;
    }
    // Gamma(0.0001, 0.0001) prior on tau, plus tau's lower-bound transform Jacobian.
    lp += -0.9999 * q[3] - 0.0001 * tau + q[3] + (0.5_f64).ln() + s.ln() + (-s).ln_1p();
    g[0] = ga;
    g[1] = gb;
    g[2] = glambda * 0.5 * s * (1.0 - s) + 1.0 - 2.0 * s;
    g[3] = 0.5 * d.x.len() as f64 + 0.0001 - tau * (0.5 * sum_sq + 0.0001);
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
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> { let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?; let data = parse_data(v)?; let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?; if ndim != 4 || got != expected_layout(&data) { return Err("dimension/layout mismatch".into()); } Ok(Bound { data, ndim }) }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 { if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); } let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) })); match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace") } }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 { let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> { if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); } let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); } let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) }; let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0) })); match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in evaluate") } }
