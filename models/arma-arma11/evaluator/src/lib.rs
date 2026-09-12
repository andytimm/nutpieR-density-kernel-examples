use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn positive_integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_f64)
        .filter(|x| x.is_finite() && *x >= 1.0 && x.fract() == 0.0)
        .ok_or_else(|| format!("{key} must be a positive integer"))?;
    Ok(x as usize)
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let t = positive_integer(&Value::Object(o.clone()), "T")?;
    let a = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if a.len() != t { return Err("y length must equal T".into()); }
    let mut y = Vec::with_capacity(t);
    for x in a {
        y.push(x.as_f64().filter(|x| x.is_finite()).ok_or("y must be finite numeric")?);
    }
    Ok(Data { y })
}
fn expected_layout() -> &'static str { "mu\nphi\ntheta\nsigma" }

// This is Stan's original recurrence with forward sensitivities. It uses the
// unconstrained q = (mu, phi, theta, log(sigma)) coordinates.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let (mu, phi, theta) = (q[0], q[1], q[2]);
    let sigma = q[3].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("non-finite sigma".into()); }
    let inv_sigma2 = 1.0 / (sigma * sigma);
    if !inv_sigma2.is_finite() { return Err("non-finite sigma precision".into()); }

    // Normal priors, with constants dropped under propto=true.
    let mut lp = -0.5 * (mu / 10.0).powi(2)
        - 0.5 * (phi / 2.0).powi(2)
        - 0.5 * (theta / 2.0).powi(2);
    g[0] = -mu / 100.0;
    g[1] = -phi / 4.0;
    g[2] = -theta / 4.0;

    // Cauchy(0, 2.5) prior on constrained sigma, and exp-transform Jacobian.
    let r = sigma / 2.5;
    let r2 = r * r;
    lp += -r2.ln_1p() + q[3];
    g[3] = 1.0 - 2.0 * r2 / (1.0 + r2);

    let mut previous_error = 0.0;
    let mut d_mu = 0.0;
    let mut d_phi = 0.0;
    let mut d_theta = 0.0;
    let mut sum_scaled_square = 0.0;
    for (i, &y) in d.y.iter().enumerate() {
        let (error, next_d_mu, next_d_phi, next_d_theta) = if i == 0 {
            (y - mu - phi * mu, -(1.0 + phi), -mu, 0.0)
        } else {
            (y - (mu + phi * d.y[i - 1] + theta * previous_error),
             -1.0 - theta * d_mu,
             -d.y[i - 1] - theta * d_phi,
             -previous_error - theta * d_theta)
        };
        let likelihood_scale = error * inv_sigma2;
        lp -= 0.5 * error * likelihood_scale + q[3];
        g[3] -= 1.0;
        g[0] -= likelihood_scale * next_d_mu;
        g[1] -= likelihood_scale * next_d_phi;
        g[2] -= likelihood_scale * next_d_theta;
        sum_scaled_square += error * likelihood_scale;
        previous_error = error;
        d_mu = next_d_mu;
        d_phi = next_d_phi;
        d_theta = next_d_theta;
    }
    g[3] += sum_scaled_square;
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
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? })
            .map_err(|_| "layout is not UTF-8")?;
        if ndim != 4 || got != expected_layout() { return Err("dimension/layout mismatch".into()); }
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
    match answer { Ok(Ok(w)) => { unsafe { *out = w }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(
    bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize,
    lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize,
) -> i32 {
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); }
        let b = unsafe { &*bound.cast::<Bound>() };
        if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); }
        let q = unsafe { slice::from_raw_parts(q, ndim) };
        let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = match eval_model(&b.data, q, g) { Ok(x) => x, Err(_) => return Ok(1) };
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }
        unsafe { *lp = value; }; Ok(0)
    }));
    match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in evaluate") }
}
