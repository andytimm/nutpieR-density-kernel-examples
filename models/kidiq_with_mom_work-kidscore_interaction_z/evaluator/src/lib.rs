use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, y: Vec<f64>, hs: Vec<f64>, iq: Vec<f64>, inter: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn vector(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let n=o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    if n < 2 { return Err("N must be at least 2 for Stan sd".into()); }
    let y=vector(&v,"kid_score",n)?; let mom_hs=vector(&v,"mom_hs",n)?; let mom_iq=vector(&v,"mom_iq",n)?;
    // Exact original transformed-data standardization: mean and sample sd.
    let standardize=|x: &[f64]| -> Result<Vec<f64>,String> { let mean=x.iter().sum::<f64>()/n as f64; let ss=x.iter().map(|z|(z-mean)*(z-mean)).sum::<f64>(); let sd=(ss/(n as f64-1.0)).sqrt(); if !sd.is_finite()||sd==0.0{return Err("predictor sd must be positive finite".into())}; Ok(x.iter().map(|z|(z-mean)/(2.0*sd)).collect()) };
    let hs=standardize(&mom_hs)?; let iq=standardize(&mom_iq)?; let inter=hs.iter().zip(&iq).map(|(a,b)|a*b).collect();
    Ok(Data {n,y,hs,iq,inter})
}
fn expected_layout(_: &Data) -> String { "beta.1
beta.2
beta.3
beta.4
sigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x|!x.is_finite()) { return Err("non-finite position".into()); }
    let sigma=q[4].exp(); if !sigma.is_finite() { return Err("sigma overflow".into()); }
    let inv=1.0/sigma; let mut lp=q[4]; // lower-bound transform Jacobian
    g.fill(0.0); g[4]=1.0;
    for i in 0..d.n { let mu=q[0]+q[1]*d.hs[i]+q[2]*d.iq[i]+q[3]*d.inter[i]; let r=(d.y[i]-mu)*inv; lp += -0.5*r*r-q[4]; let w=r*inv; g[0]+=w; g[1]+=w*d.hs[i]; g[2]+=w*d.iq[i]; g[3]+=w*d.inter[i]; g[4]+=r*r-1.0; }
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
        let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? })
            .map_err(|_| "layout is not UTF-8")?;
        if ndim != 5 || got != want { return Err("dimension/layout mismatch".into()); }
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