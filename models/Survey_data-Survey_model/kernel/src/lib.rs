use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { nmax: usize, k: Vec<usize>, nmin: usize }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonnegative_integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.as_u64().ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let nmax = nonnegative_integer(o.get("nmax").ok_or("missing nmax")?, "nmax")?;
    let m = nonnegative_integer(o.get("m").ok_or("missing m")?, "m")?;
    let a = o.get("k").and_then(Value::as_array).ok_or("k must be an integer array")?;
    if a.len() != m { return Err("k length must equal m".into()); }
    let mut k = Vec::with_capacity(m);
    for x in a { let v = nonnegative_integer(x, "k")?; if v > nmax { return Err("k must be <= nmax".into()); } k.push(v); }
    let nmin = k.iter().copied().max().unwrap_or(0); // original transformed-data nmin = max(k)
    Ok(Data { nmax, k, nmin })
}
fn expected_layout(_: &Data) -> String { "theta".into() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn log_choose(n: usize, k: usize) -> f64 {
    // Equivalent to log(n choose k), evaluated without factorial allocation.
    let r = k.min(n-k);
    let mut ans = 0.0;
    for j in 1..=r { ans += ((n-r+j) as f64).ln() - (j as f64).ln(); }
    ans
}
fn eval_model(d: &Data, _w: &mut Workspace, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let theta = sigmoid(q[0]);
    let log_theta = -((-q[0]).exp()).ln_1p();
    let log1m_theta = -(q[0].exp()).ln_1p();
    // Streaming log-sum-exp and its derivative remove the per-component
    // workspace write/read passes. Every original component is still evaluated.
    let mut max_term = f64::NEG_INFINITY;
    let mut sum = 0.0;
    let mut weighted_dq = 0.0;
    for n in d.nmin.max(1)..=d.nmax {
        let mut term = -((d.nmax as f64).ln());
        let mut dq = 0.0;
        for &ki in &d.k {
            term += log_choose(n, ki) + (ki as f64) * log_theta + ((n-ki) as f64) * log1m_theta;
            dq += (ki as f64) - (n as f64) * theta;
        }
        if term > max_term {
            let scale = (max_term - term).exp();
            sum = sum * scale + 1.0;
            weighted_dq = weighted_dq * scale + dq;
            max_term = term;
        } else {
            let weight = (term - max_term).exp();
            sum += weight;
            weighted_dq += weight * dq;
        }
    }
    if sum == 0.0 { return Err("empty mixture support".into()); }
    let lse = max_term + sum.ln();
    let weighted_dq = weighted_dq / sum;
    // theta = inv_logit(q): include lower/upper-bound transform Jacobian.
    g[0] = weighted_dq + 1.0 - 2.0*theta;
    Ok(lse + log_theta + log1m_theta)
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
        let b = unsafe { &*bound.cast::<Bound>() };
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
        let w = unsafe { &mut *workspace.cast::<Workspace>() };
        let value = eval_model(&b.data, w, q, g)?;
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }
        unsafe { *lp = value; }; Ok(0)
    }));
    match answer {
        Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density kernel panic in evaluate"),
    }
}