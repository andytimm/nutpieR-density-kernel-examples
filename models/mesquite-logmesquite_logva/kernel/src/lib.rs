use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data {
    n: usize,
    log_weight: Vec<f64>,
    log_canopy_volume: Vec<f64>,
    log_canopy_area: Vec<f64>,
    group: Vec<f64>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn finite_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} must have length N")); }
    a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>()
        .filter(|x| x.iter().all(|v| v.is_finite()))
        .ok_or_else(|| format!("{key} must contain finite numeric values"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_value = finite_num(&Value::Object(o.clone()), "N")?;
    if n_value < 0. || n_value.fract() != 0. || n_value > usize::MAX as f64 { return Err("N must be a nonnegative integer".into()); }
    let n = n_value as usize;
    let weight = finite_vec(o, "weight", n)?;
    let diam1 = finite_vec(o, "diam1", n)?;
    let diam2 = finite_vec(o, "diam2", n)?;
    let height = finite_vec(o, "canopy_height", n)?;
    let group = finite_vec(o, "group", n)?;
    if weight.iter().chain(diam1.iter()).chain(diam2.iter()).chain(height.iter()).any(|x| *x <= 0.) {
        return Err("weight, diam1, diam2, and canopy_height must be > 0".into());
    }
    // These are precisely the original Stan transformed-data calculations.
    let mut log_weight = Vec::with_capacity(n);
    let mut log_canopy_volume = Vec::with_capacity(n);
    let mut log_canopy_area = Vec::with_capacity(n);
    for i in 0..n {
        log_weight.push(weight[i].ln());
        log_canopy_volume.push((diam1[i] * diam2[i] * height[i]).ln());
        log_canopy_area.push((diam1[i] * diam2[i]).ln());
    }
    Ok(Data { n, log_weight, log_canopy_volume, log_canopy_area, group })
}
fn expected_layout(_d: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let sigma = q[4].exp();
    if !sigma.is_finite() { return Err("sigma transform overflow".into()); }
    let inv_sigma_sq = 1.0 / (sigma * sigma);
    let mut lp = q[4]; // lower-bound transform Jacobian
    let mut grad_log_sigma = 1.0;
    g.fill(0.0);
    for i in 0..d.n {
        let mean = q[0] + q[1] * d.log_canopy_volume[i] + q[2] * d.log_canopy_area[i] + q[3] * d.group[i];
        let r = d.log_weight[i] - mean;
        let scaled_r = r / sigma;
        // normal_lpdf with propto=true retains parameter-dependent -log(sigma).
        lp += -q[4] - 0.5 * scaled_r * scaled_r;
        let coeff = r * inv_sigma_sq;
        g[0] += coeff;
        g[1] += coeff * d.log_canopy_volume[i];
        g[2] += coeff * d.log_canopy_area[i];
        g[3] += coeff * d.group[i];
        grad_log_sigma += -1.0 + scaled_r * scaled_r;
    }
    g[4] = grad_log_sigma;
    Ok(lp)
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
    if out.is_null() { return fatal(err, cap, "null bound output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?; let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != 5 || got != want { return Err("dimension/layout mismatch".into()); }
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
