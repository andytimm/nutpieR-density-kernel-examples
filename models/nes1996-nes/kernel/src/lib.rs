use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data {
    y: Vec<f64>, real_ideo: Vec<f64>, race_adj: Vec<f64>, educ1: Vec<f64>,
    gender: Vec<f64>, income: Vec<f64>, age: Vec<i64>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must contain finite numerics"))
}
fn vector(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|v| number(v, key)).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_i64).filter(|x| *x >= 0).ok_or("N must be a nonnegative integer")? as usize;
    let age_v = o.get("age_discrete").and_then(Value::as_array).ok_or("age_discrete must be an array")?;
    if age_v.len() != n { return Err("age_discrete has wrong length".into()); }
    let age: Result<Vec<i64>, String> = age_v.iter().map(|v| v.as_i64().filter(|x| (1..=4).contains(x)).ok_or_else(|| "age_discrete must be integers 1 through 4".into())).collect();
    Ok(Data { y: vector(o,"partyid7",n)?, real_ideo: vector(o,"real_ideo",n)?, race_adj: vector(o,"race_adj",n)?, educ1: vector(o,"educ1",n)?, gender: vector(o,"gender",n)?, income: vector(o,"income",n)?, age: age? })
}
fn expected_layout(_d: &Data) -> String {
    (1..=9).map(|i| format!("beta.{i}")).chain(std::iter::once("sigma".into())).collect::<Vec<_>>().join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let u = q[9];
    // lower=0 transform: sigma=exp(u), followed by its log-Jacobian u.
    let sigma = u.exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform outside finite domain".into()); }
    let inv_sigma2 = 1.0 / (sigma * sigma);
    if !inv_sigma2.is_finite() { return Err("sigma transform outside finite domain".into()); }
    g.fill(0.0);
    let mut ss = 0.0;
    for i in 0..d.y.len() {
        // This reproduces the transformed-data age indicator work per evaluation.
        let a = d.age[i];
        let x = [1.0, d.real_ideo[i], d.race_adj[i], (a == 2) as u8 as f64,
                 (a == 3) as u8 as f64, (a == 4) as u8 as f64,
                 d.educ1[i], d.gender[i], d.income[i]];
        let mut mu = 0.0;
        for j in 0..9 { mu += q[j] * x[j]; }
        let r = d.y[i] - mu;
        ss += r * r * inv_sigma2;
        let scale = r * inv_sigma2;
        for j in 0..9 { g[j] += x[j] * scale; }
    }
    // normal_lpdf<propto>: -1/2 sum squared residuals/sigma^2 - N log(sigma).
    // Add the lower-bound transform log-Jacobian u.
    g[9] = ss - d.y.len() as f64 + 1.0;
    Ok(-0.5 * ss - (d.y.len() as f64 - 1.0) * u)
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

#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }

#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(
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
        if ndim != 10 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer {
        Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }
        Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "producer panic in bind"),
    }
}

#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() {
        unsafe { drop(Box::from_raw(bound.cast::<Bound>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(
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
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in workspace"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() {
        unsafe { drop(Box::from_raw(w.cast::<Workspace>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(
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
        Err(_) => fatal(err, cap, "producer panic in evaluate"),
    }
}