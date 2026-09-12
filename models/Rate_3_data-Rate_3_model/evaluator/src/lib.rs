use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies of the supplied Stan integer data. Workspace is private per chain.
struct Data { n1: u32, n2: u32, k1: u32, k2: u32 }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<u32, String> {
    let x = v.get(key).and_then(Value::as_i64)
        .ok_or_else(|| format!("{key} must be an integer"))?;
    u32::try_from(x).map_err(|_| format!("{key} is outside Stan integer range"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let v = Value::Object(o.clone());
    let n1 = integer(&v, "n1")?;
    let n2 = integer(&v, "n2")?;
    let k1 = integer(&v, "k1")?;
    let k2 = integer(&v, "k2")?;
    if n1 == 0 || n2 == 0 { return Err("n1 and n2 must be >= 1".into()); }
    if k1 > n1 || k2 > n2 { return Err("binomial count exceeds trials".into()); }
    Ok(Data { n1, n2, k1, k2 })
}
fn expected_layout(_: &Data) -> String { "theta".into() }

fn log_inv_logit(x: f64) -> f64 {
    if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() }
}
fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let x = q[0];
    if !x.is_finite() { return Err("non-finite unconstrained position".into()); }
    // Stan: beta(1,1) contributes no propto terms. The two binomial terms retain
    // theta-dependent powers; the final two terms are the lower/upper logistic Jacobian.
    let log_theta = log_inv_logit(x);
    let log_one_minus_theta = log_inv_logit(-x);
    let theta = inv_logit(x);
    // Keep the original two observed binomial terms separate: no data-only
    // sufficient totals are formed at binding or evaluation time.
    let term1 = f64::from(d.k1) * log_theta + f64::from(d.n1 - d.k1) * log_one_minus_theta;
    let term2 = f64::from(d.k2) * log_theta + f64::from(d.n2 - d.k2) * log_one_minus_theta;
    let grad1 = f64::from(d.k1) * (1.0 - theta) - f64::from(d.n1 - d.k1) * theta;
    let grad2 = f64::from(d.k2) * (1.0 - theta) - f64::from(d.n2 - d.k2) * theta;
    // The lower/upper bounded transform adds log(theta) + log(1 - theta).
    g[0] = grad1 + grad2 + (1.0 - 2.0 * theta);
    Ok(term1 + term2 + log_theta + log_one_minus_theta)
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
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 {
    unsafe { put_error(p, cap, s.as_ref()) }; 2
}

#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }

#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(
    json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char,
    layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize,
) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? })
            .map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?;
        let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? })
            .map_err(|_| "layout is not UTF-8")?;
        if ndim != 1 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer {
        Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in bind"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() {
        unsafe { drop(Box::from_raw(bound.cast::<Bound>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(
    bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize,
) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        if bound.is_null() { return Err("null bound handle".into()); }
        Ok(Box::into_raw(Box::new(Workspace)).cast())
    }));
    match answer {
        Ok(Ok(w)) => { unsafe { *out = w; }; 0 }
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() {
        unsafe { drop(Box::from_raw(w.cast::<Workspace>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(
    bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize,
    lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize,
) -> i32 {
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() {
            return Err("null evaluation handle/output".into());
        }
        let b = unsafe { &*bound.cast::<Bound>() };
        if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); }
        let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } };
        let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = eval_model(&b.data, q, g)?;
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }
        unsafe { *lp = value; }; Ok(0)
    }));
    match answer {
        Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density evaluator panic in evaluate"),
    }
}
