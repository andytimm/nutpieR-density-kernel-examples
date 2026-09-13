use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data {
    n_child: usize,
    n_ind: usize,
    ncat: Vec<usize>,
    gamma: Vec<f64>, // row-major n_ind x 4, as supplied by Stan JSON
    delta: Vec<f64>,
    grade: Vec<i32>, // row-major n_child x n_ind
}
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
fn num_array<'a>(o: &'a serde_json::Map<String, Value>, key: &str, n: usize) -> Result<&'a Vec<Value>, String> {
    let x = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if x.len() != n { return Err(format!("{key} has wrong length")); }
    Ok(x)
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_child = integer(&Value::Object(o.clone()), "nChild")?;
    let n_ind = integer(&Value::Object(o.clone()), "nInd")?;
    if n_child == 0 || n_ind == 0 { return Err("nChild and nInd must be positive".into()); }
    let ncat_v = num_array(o, "ncat", n_ind)?;
    let mut ncat = Vec::with_capacity(n_ind);
    for x in ncat_v {
        let k = x.as_f64().filter(|z| z.is_finite() && *z >= 2.0 && z.fract() == 0.0 && *z <= 5.0)
            .ok_or("ncat values must be integers from 2 through 5")? as usize;
        ncat.push(k);
    }
    let delta_v = num_array(o, "delta", n_ind)?;
    let mut delta = Vec::with_capacity(n_ind);
    for x in delta_v { delta.push(x.as_f64().filter(|z| z.is_finite()).ok_or("delta must be finite numeric")?); }
    let gamma_rows = num_array(o, "gamma", n_ind)?;
    let mut gamma = Vec::with_capacity(n_ind * 4);
    for row in gamma_rows {
        let row = row.as_array().ok_or("gamma must be a matrix")?;
        if row.len() != 4 { return Err("gamma must have 4 columns".into()); }
        for x in row { gamma.push(x.as_f64().filter(|z| z.is_finite()).ok_or("gamma must be finite numeric")?); }
    }
    let grade_rows = num_array(o, "grade", n_child)?;
    let mut grade = Vec::with_capacity(n_child * n_ind);
    for row in grade_rows {
        let row = row.as_array().ok_or("grade must be a matrix")?;
        if row.len() != n_ind { return Err("grade has wrong column count".into()); }
        for (j, x) in row.iter().enumerate() {
            let y = x.as_f64().filter(|z| z.is_finite() && z.fract() == 0.0 && *z >= -1.0 && *z <= i32::MAX as f64)
                .ok_or("grade must be integer")? as i32;
            if y != -1 && (y < 1 || y as usize > ncat[j]) { return Err("grade outside category range".into()); }
            grade.push(y);
        }
    }
    Ok(Data { n_child, n_ind, ncat, gamma, delta, grade })
}
fn expected_layout(d: &Data) -> String {
    (1..=d.n_child).map(|i| format!("theta.{i}")).collect::<Vec<_>>().join("\n")
}
#[inline] fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let mut lp = 0.0;
    for i in 0..d.n_child {
        let theta = q[i];
        // normal(0, 36), with propto=true: retain only the quadratic term.
        lp -= 0.5 * theta * theta / (36.0 * 36.0);
        g[i] = -theta / (36.0 * 36.0);
        for j in 0..d.n_ind {
            let y = d.grade[i * d.n_ind + j];
            if y == -1 { continue; }
            let k = d.ncat[j];
            let delta = d.delta[j];
            let base = j * 4;
            let cat = y as usize;
            // Q_r = inv_logit(delta * (theta - gamma_r)), r is one-based.
            if cat == 1 {
                let z = delta * (theta - d.gamma[base]);
                // log(1-Q_1) = log_inv_logit(-z), evaluated stably.
                lp -= if z > 0.0 { z + (-z).exp().ln_1p() } else { z.exp().ln_1p() };
                g[i] -= delta * sigmoid(z);
            } else if cat == k {
                let z = delta * (theta - d.gamma[base + k - 2]);
                // log(Q_(K-1)) = log_inv_logit(z).
                lp -= if z > 0.0 { (-z).exp().ln_1p() } else { -z + z.exp().ln_1p() };
                g[i] += delta * (1.0 - sigmoid(z));
            } else {
                let zp = delta * (theta - d.gamma[base + cat - 2]);
                let zc = delta * (theta - d.gamma[base + cat - 1]);
                let qp = sigmoid(zp);
                let qc = sigmoid(zc);
                let p = qp - qc;
                if !(p > 0.0) || !p.is_finite() { return Ok(f64::NEG_INFINITY); }
                lp += p.ln();
                g[i] += delta * (qp * (1.0 - qp) - qc * (1.0 - qc)) / p;
            }
        }
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> { let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?; let data = parse_data(v)?; let want = expected_layout(&data); let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?; if ndim != data.n_child || got != want { return Err("dimension/layout mismatch".into()); } Ok(Bound { data, ndim }) }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 { if out.is_null() { return fatal(err, cap, "null workspace output"); } unsafe { *out = std::ptr::null_mut(); } let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> { if bound.is_null() { return Err("null bound handle".into()); } Ok(Box::into_raw(Box::new(Workspace)).cast()) })); match answer { Ok(Ok(w)) => { unsafe { *out = w; }; 0 }, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in workspace") } }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() { unsafe { drop(Box::from_raw(w.cast::<Workspace>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize, lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize) -> i32 { let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> { if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() { return Err("null evaluation handle/output".into()); } let b = unsafe { &*bound.cast::<Bound>() }; if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); } let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } }; let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) }; let value = eval_model(&b.data, q, g)?; if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); } unsafe { *lp = value; }; Ok(0) })); match answer { Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in evaluate") } }
