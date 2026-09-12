use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n1: u64, n2: u64, k1: u64, k2: u64 }
struct Bound { data: Data, ndim: usize }
struct Workspace; // private, per-chain workspace

fn nonnegative_integer(o: &serde_json::Map<String, Value>, key: &str) -> Result<u64, String> {
    o.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn positive_integer(o: &serde_json::Map<String, Value>, key: &str) -> Result<u64, String> {
    let x = nonnegative_integer(o, key)?;
    if x == 0 { Err(format!("{key} must be positive")) } else { Ok(x) }
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n1 = positive_integer(o, "n1")?;
    let n2 = positive_integer(o, "n2")?;
    let k1 = nonnegative_integer(o, "k1")?;
    let k2 = nonnegative_integer(o, "k2")?;
    if k1 > n1 || k2 > n2 { return Err("counts must not exceed trials".into()); }
    Ok(Data { n1, n2, k1, k2 })
}
fn expected_layout(_: &Data) -> String { "theta".into() }

// theta is declared real<lower=0, upper=1>, so q = logit(theta).
// With propto=TRUE the beta(1,1) and binomial combinatorial constants drop.
// The transform Jacobian is retained: log(theta) + log1p(-theta).
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let z = q[0];
    if !z.is_finite() { return Err("non-finite position".into()); }
    let theta = if z >= 0.0 { 1.0 / (1.0 + (-z).exp()) } else { z.exp() / (1.0 + z.exp()) };
    let log_theta = if z >= 0.0 { -((-z).exp().ln_1p()) } else { z - z.exp().ln_1p() };
    let log_one_minus_theta = if z >= 0.0 { -z - (-z).exp().ln_1p() } else { -z.exp().ln_1p() };
    // Keep the two original Stan binomial statements separate. This is not a
    // bind-time or per-evaluation sufficient-statistic reduction.
    let n1 = d.n1 as f64;
    let n2 = d.n2 as f64;
    let k1 = d.k1 as f64;
    let k2 = d.k2 as f64;
    let lp1 = k1 * log_theta + (n1 - k1) * log_one_minus_theta;
    let lp2 = k2 * log_theta + (n2 - k2) * log_one_minus_theta;
    let grad1 = k1 - n1 * theta;
    let grad2 = k2 - n2 * theta;
    // Add the lower/upper-bound logit transform Jacobian separately.
    let jacobian = log_theta + log_one_minus_theta;
    let jacobian_grad = 1.0 - 2.0 * theta;
    g[0] = grad1 + grad2 + jacobian_grad;
    Ok(lp1 + lp2 + jacobian)
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
        Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density evaluator panic in bind"),
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