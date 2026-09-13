use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Values below reproduce the Stan transformed-data statements at binding.
// Bound is immutable; Workspace is allocated independently for each chain.
struct Data { log_weight: Vec<f64>, log_canopy_volume: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn numeric_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|v| v.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_i64).ok_or("N must be an integer")?;
    if n < 0 { return Err("N is out of range".into()); }
    let n = n as usize;
    let weight = numeric_array(o, "weight", n)?;
    let diam1 = numeric_array(o, "diam1", n)?;
    let diam2 = numeric_array(o, "diam2", n)?;
    let canopy_height = numeric_array(o, "canopy_height", n)?;
    let mut log_weight = Vec::with_capacity(n);
    let mut log_canopy_volume = Vec::with_capacity(n);
    for i in 0..n {
        // Stan's transformed data evaluates these same logs, so their domains are required.
        if weight[i] <= 0.0 || diam1[i] <= 0.0 || diam2[i] <= 0.0 || canopy_height[i] <= 0.0 {
            return Err("weight, diam1, diam2, and canopy_height must be positive".into());
        }
        let y = weight[i].ln();
        let x = (diam1[i] * diam2[i] * canopy_height[i]).ln();
        if !y.is_finite() || !x.is_finite() { return Err("transformed data is non-finite".into()); }
        log_weight.push(y);
        log_canopy_volume.push(x);
    }
    Ok(Data { log_weight, log_canopy_volume })
}
fn expected_layout(_d: &Data) -> String { "beta.1\nbeta.2\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|v| !v.is_finite()) { return Err("non-finite position".into()); }
    let beta0 = q[0];
    let beta1 = q[1];
    let log_sigma = q[2];
    let sigma_inv = (-log_sigma).exp();
    if !sigma_inv.is_finite() { return Err("sigma transform is non-finite".into()); }
    let inv_var = sigma_inv * sigma_inv;
    if !inv_var.is_finite() { return Err("sigma inverse square is non-finite".into()); }
    let mut lp = log_sigma; // log-Jacobian for real<lower=0> sigma = exp(log_sigma)
    let mut gbeta0 = 0.0;
    let mut gbeta1 = 0.0;
    let mut glog_sigma = 1.0;
    for (&y, &x) in d.log_weight.iter().zip(d.log_canopy_volume.iter()) {
        let residual = y - (beta0 + beta1 * x);
        let scaled = residual * sigma_inv;
        lp += -0.5 * scaled * scaled - log_sigma; // normal<propto=true>: retain -log(sigma)
        let common = residual * inv_var;
        gbeta0 += common;
        gbeta1 += common * x;
        glog_sigma += scaled * scaled - 1.0;
    }
    g[0] = gbeta0; g[1] = gbeta1; g[2] = glog_sigma;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> { let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?; let data = parse_data(v)?; let want = expected_layout(&data); let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?; if ndim != 3 || got != want { return Err("dimension/layout mismatch".into()); } Ok(Bound { data, ndim }) }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 { if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); } let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) })); match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in workspace") } }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 { let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> { if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); } let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); } let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) }; let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0) })); match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in evaluate") } }
