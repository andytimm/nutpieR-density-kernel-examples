use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county: Vec<usize>, log_radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = finite_num(v, key)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 { Err(format!("{key} must be a nonnegative integer")) }
    else { Ok(x as usize) }
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let obj = Value::Object(o.clone());
    let n = integer(&obj, "N")?;
    let j = integer(&obj, "J")?;
    let c = obj.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    let y = obj.get("log_radon").and_then(Value::as_array).ok_or("log_radon must be an array")?;
    if c.len() != n || y.len() != n { return Err("data vector length mismatch".into()); }
    let mut county = Vec::with_capacity(n);
    let mut log_radon = Vec::with_capacity(n);
    for (i, x) in c.iter().enumerate() {
        let k = x.as_u64().ok_or_else(|| format!("county_idx[{i}] must be integer"))? as usize;
        if k == 0 || k > j { return Err(format!("county_idx[{i}] outside 1..J")); }
        county.push(k - 1);
    }
    for (i, x) in y.iter().enumerate() {
        log_radon.push(x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("log_radon[{i}] must be finite"))?);
    }
    Ok(Data { n, j, county, log_radon })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.j).map(|i| format!("alpha_raw.{i}")).collect();
    names.extend(["mu_alpha".into(), "sigma_alpha".into(), "sigma_y".into()]);
    names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let mu_i = d.j;
    let sa_i = d.j + 1;
    let sy_i = d.j + 2;
    let mu = q[mu_i];
    let sigma_alpha = q[sa_i].exp();
    let sigma_y = q[sy_i].exp();
    if !mu.is_finite() || !sigma_alpha.is_finite() || !sigma_y.is_finite() || sigma_alpha == 0.0 || sigma_y == 0.0 { return Err("non-finite transformed parameter".into()); }
    g.fill(0.0);
    let mut lp = q[sa_i] + q[sy_i]; // lower-bound transform Jacobians
    let mut d_mu = -mu / 100.0;
    let mut d_sa = -sigma_alpha;
    let mut d_sy = -sigma_y;
    lp += -0.5 * (mu / 10.0) * (mu / 10.0) - 0.5 * sigma_alpha * sigma_alpha - 0.5 * sigma_y * sigma_y;
    for k in 0..d.j { lp += -0.5 * q[k] * q[k]; g[k] = -q[k]; }
    // The explicit normal_lpdf in the Stan source retains its -0.5 log(2*pi) term.
    const LOG_SQRT_2PI: f64 = 0.91893853320467274178032973640562;
    let inv_sy = 1.0 / sigma_y;
    for n in 0..d.n {
        let k = d.county[n];
        let alpha = mu + sigma_alpha * q[k];
        let z = (d.log_radon[n] - alpha) * inv_sy;
        lp += -0.5 * z * z - q[sy_i] - LOG_SQRT_2PI;
        let d_alpha = z * inv_sy;
        g[k] += d_alpha * sigma_alpha;
        d_mu += d_alpha;
        d_sa += d_alpha * q[k];
        d_sy += (z * z - 1.0) * inv_sy;
    }
    g[mu_i] = d_mu;
    g[sa_i] = d_sa * sigma_alpha + 1.0;
    g[sy_i] = d_sy * sigma_y + 1.0;
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
        if ndim != data.j + 3 || got != want { return Err("dimension/layout mismatch".into()); }
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
