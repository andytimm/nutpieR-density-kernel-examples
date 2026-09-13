use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: u64, k: u64 }
struct Bound { data: Data, ndim: usize }
struct Workspace; // private, per chain

fn integer(v: &Value, key: &str) -> Result<u64, String> {
    let x = v.get(key).and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    Ok(x)
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = integer(&Value::Object(o.clone()), "n")?;
    let k = integer(&Value::Object(o.clone()), "k")?;
    if n < 1 { return Err("n must be >= 1".into()); }
    if k > n { return Err("k must be <= n".into()); }
    Ok(Data { n, k })
}
fn expected_layout(_d: &Data) -> String { "theta".into() }

// Stan: theta ~ beta(1, 1); k ~ binomial(n, theta).
// With propto=true the beta(1,1) prior contributes zero. For the bounded
// parameter theta=inv_logit(q), add log|d theta/dq|.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let x = q[0];
    if !x.is_finite() { return Err("non-finite unconstrained position".into()); }
    let theta = if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { x.exp() / (1.0 + x.exp()) };
    // Stable log(inv_logit(x)) and log1m(inv_logit(x)); the exp is required.
    let log_theta = -((-x).max(0.0) + (-(-x).abs()).exp().ln_1p());
    let log1m_theta = -(x.max(0.0) + (-x.abs()).exp().ln_1p());
    let k = d.k as f64;
    let n = d.n as f64;
    // Binomial propto likelihood plus the lower/upper-bound Jacobian.
    let value = (k + 1.0) * log_theta + (n - k + 1.0) * log1m_theta;
    g[0] = k + 1.0 - (n + 2.0) * theta;
    Ok(value)
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

#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }

#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(
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
        Err(_) => fatal(err, cap, "density kernel panic in bind"),
    }
}

#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() {
        unsafe { drop(Box::from_raw(bound.cast::<Bound>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(
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
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in workspace"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() {
        unsafe { drop(Box::from_raw(w.cast::<Workspace>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(
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
        Err(_) => fatal(err, cap, "density kernel panic in evaluate"),
    }
}