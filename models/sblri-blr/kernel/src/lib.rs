use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, d: usize, x: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonnegative_usize(v: &Value, key: &str) -> Result<usize, String> {
    let n = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(n).map_err(|_| format!("{key} is too large"))
}
fn finite(v: &Value, what: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{what} must be finite numeric"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = nonnegative_usize(&v, "N")?;
    let d = nonnegative_usize(&v, "D")?;
    let yv = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if yv.len() != n { return Err("y length does not match N".into()); }
    let mut y = Vec::with_capacity(n);
    for z in yv { y.push(finite(z, "y value")?); }
    let xv = o.get("X").and_then(Value::as_array).ok_or("X must be an array")?;
    if xv.len() != n { return Err("X row count does not match N".into()); }
    let len = n.checked_mul(d).ok_or("X dimensions overflow")?;
    let mut x = Vec::with_capacity(len);
    for row in xv {
        let row = row.as_array().ok_or("X must be a two-dimensional array")?;
        if row.len() != d { return Err("X column count does not match D".into()); }
        for z in row { x.push(finite(z, "X value")?); }
    }
    Ok(Data { n, d, x, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.d + 1);
    for j in 1..=d.d { names.push(format!("beta.{j}")); }
    names.push("sigma".into());
    names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    // Stan: normal_lpdf(beta | 0, 10), normal_lpdf(sigma | 0, 10),
    // normal_lpdf(y | X * beta, sigma), propto=true, jacobian=true.
    let beta = &q[..d.d];
    let z = q[d.d];
    let sigma = z.exp();
    if !sigma.is_finite() { return Err("sigma transform is non-finite".into()); }
    let inv_sigma2 = 1.0 / (sigma * sigma);
    if !inv_sigma2.is_finite() { return Err("sigma scale is non-finite".into()); }
    for j in 0..d.d { g[j] = -beta[j] / 100.0; }
    let mut lp = 0.0;
    let mut rss = 0.0;
    for i in 0..d.n {
        let row = &d.x[i * d.d..(i + 1) * d.d];
        let mut eta = 0.0;
        for j in 0..d.d { eta += row[j] * beta[j]; }
        let residual = d.y[i] - eta;
        rss += residual * residual;
        let scale = residual * inv_sigma2;
        for j in 0..d.d { g[j] += row[j] * scale; }
    }
    for b in beta { lp += -0.5 * b * b / 100.0; }
    lp += -0.5 * sigma * sigma / 100.0;
    lp += -(d.n as f64) * z - 0.5 * rss * inv_sigma2;
    // Explicit Stan normal_lpdf calls retain their normalizing terms here.
    let normal_count = (d.n + d.d + 1) as f64;
    lp += -0.5 * normal_count * std::f64::consts::TAU.ln()
        - ((d.d + 1) as f64) * 10.0_f64.ln();
    g[d.d] = 1.0 - d.n as f64 - sigma * sigma / 100.0 + rss * inv_sigma2;
    Ok(lp + z)
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
    match catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let data = parse_data(serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?)?;
        let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != data.d + 1 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    })) { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); }
    match catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { Err("null bound handle".into()) } else { Ok(Box::into_raw(Box::new(Workspace)).cast()) } })) { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err,cap,s), Err(_) => fatal(err,cap,"producer panic in workspace") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); }
        let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); }
        let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0)
    })) { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err,cap,s), Err(_) => fatal(err,cap,"producer panic in evaluate") }
}
