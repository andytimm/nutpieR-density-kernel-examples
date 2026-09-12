use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

/// Immutable copies of the supplied Stan data. `mom_work` remains in its
/// original integer form; its transformed-data indicators are evaluated in
/// the likelihood loop, exactly as in the Stan program.
struct Data { n: usize, kid_score: Vec<f64>, mom_work: Vec<i64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_i = o.get("N").and_then(Value::as_i64).ok_or("N must be an integer")?;
    if n_i < 0 { return Err("N must be nonnegative".into()); }
    let n = n_i as usize;
    let ys = o.get("kid_score").and_then(Value::as_array).ok_or("kid_score must be an array")?;
    let works = o.get("mom_work").and_then(Value::as_array).ok_or("mom_work must be an array")?;
    if ys.len() != n || works.len() != n { return Err("data array length does not match N".into()); }
    let mut kid_score = Vec::with_capacity(n);
    let mut mom_work = Vec::with_capacity(n);
    for (i, y) in ys.iter().enumerate() {
        kid_score.push(y.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("kid_score[{}] must be finite numeric", i + 1))?);
    }
    for (i, w) in works.iter().enumerate() {
        mom_work.push(w.as_i64().ok_or_else(|| format!("mom_work[{}] must be an integer", i + 1))?);
    }
    Ok(Data { n, kid_score, mom_work })
}
fn expected_layout(_d: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nsigma".into() }

/// Stan: kid_score ~ normal(beta[1] + beta[2]*work2 + beta[3]*work3 +
/// beta[4]*work4, sigma), with `propto=true,jacobian=true`.
/// `sigma = exp(q[4])`; sigma is a parameter, so normal_lpdf retains -log(sigma),
/// and the lower-bound transform contributes +q[4].
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let sigma = q[4].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform outside finite range".into()); }
    let inv_sigma = 1.0 / sigma;
    let inv_sigma_sq = inv_sigma * inv_sigma;
    g.fill(0.0);
    let mut ss = 0.0;
    for i in 0..d.n {
        // These are the original transformed-data definitions, retained as
        // per-observation work rather than a data-only reduced statistic.
        let w = d.mom_work[i];
        let eta = q[0] + q[1] * if w == 2 { 1.0 } else { 0.0 }
                       + q[2] * if w == 3 { 1.0 } else { 0.0 }
                       + q[3] * if w == 4 { 1.0 } else { 0.0 };
        let residual = d.kid_score[i] - eta;
        let scaled = residual * inv_sigma_sq;
        ss += residual * scaled;
        g[0] += scaled;
        if w == 2 { g[1] += scaled; }
        if w == 3 { g[2] += scaled; }
        if w == 4 { g[3] += scaled; }
    }
    // The normal constant is parameter independent and dropped by propto.
    // Chain rule for log(sigma): ss - N; plus lower-bound exp Jacobian: +1.
    g[4] = ss - d.n as f64 + 1.0;
    Ok(-0.5 * ss - d.n as f64 * q[4] + q[4])
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
    if out.is_null() { return fatal(err, cap, "null bound output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> { let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?; let data = parse_data(v)?; let want = expected_layout(&data); let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?; if ndim != 5 || got != want { return Err("dimension/layout mismatch".into()); } Ok(Bound { data, ndim }) }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 { if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); } let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) })); match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in workspace") } }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 { let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> { if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); } let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || q.is_null() { return Err("evaluation dimension".into()); } let q = unsafe { slice::from_raw_parts(q, ndim) }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) }; let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0) })); match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in evaluate") } }
