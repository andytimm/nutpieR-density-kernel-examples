use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data {
    n: usize,
    partyid7: Vec<f64>, real_ideo: Vec<f64>, race_adj: Vec<f64>, educ1: Vec<f64>,
    gender: Vec<f64>, income: Vec<f64>, age_discrete: Vec<i64>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} must have length N")); }
    a.iter().map(|x| x.as_f64().filter(|v| v.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn integer_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<i64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an integer array"))?;
    if a.len() != n { return Err(format!("{key} must have length N")); }
    a.iter().map(|x| x.as_i64().filter(|v| (1..=4).contains(v)).ok_or_else(|| format!("{key} must contain integers 1 through 4"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).filter(|&x| x <= usize::MAX as u64).ok_or("N must be a nonnegative integer")? as usize;
    Ok(Data { n,
        partyid7: finite_vec(o, "partyid7", n)?, real_ideo: finite_vec(o, "real_ideo", n)?,
        race_adj: finite_vec(o, "race_adj", n)?, educ1: finite_vec(o, "educ1", n)?,
        gender: finite_vec(o, "gender", n)?, income: finite_vec(o, "income", n)?,
        age_discrete: integer_vec(o, "age_discrete", n)?,
    })
}
fn expected_layout(_: &Data) -> String {
    (1..=9).map(|i| format!("beta.{i}")).chain(std::iter::once("sigma".to_owned())).collect::<Vec<_>>().join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    g.fill(0.0);
    let sigma = q[9].exp();
    let inv_sigma = 1.0 / sigma;
    let inv_sigma_sq = inv_sigma * inv_sigma;
    let mut lp = -(d.n as f64) * q[9];
    let mut grad_sigma = -(d.n as f64) + 1.0; // normal scale plus log-Jacobian
    for n in 0..d.n {
        // These comparisons reproduce the Stan transformed-data age factors.
        let a30 = if d.age_discrete[n] == 2 { 1.0 } else { 0.0 };
        let a45 = if d.age_discrete[n] == 3 { 1.0 } else { 0.0 };
        let a65 = if d.age_discrete[n] == 4 { 1.0 } else { 0.0 };
        let x = [1.0, d.real_ideo[n], d.race_adj[n], a30, a45, a65, d.educ1[n], d.gender[n], d.income[n]];
        let mut eta = 0.0;
        for j in 0..9 { eta += q[j] * x[j]; }
        let r = d.partyid7[n] - eta;
        let z = r * inv_sigma;
        lp += -0.5 * z * z;
        let scale = r * inv_sigma_sq;
        for j in 0..9 { g[j] += scale * x[j]; }
        grad_sigma += z * z;
    }
    lp += q[9];
    g[9] = grad_sigma;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } }
}
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?; let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != 10 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err,cap,s), Err(_) => fatal(err,cap,"density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) { let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err,cap,"null workspace output"); } unsafe { *out=std::ptr::null_mut(); }
    let answer=catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void,String> { if bound.is_null(){return Err("null bound handle".into())} Ok(Box::into_raw(Box::new(Workspace)).cast()) }));
    match answer { Ok(Ok(w))=>{unsafe{*out=w};0}, Ok(Err(s))=>fatal(err,cap,s), Err(_)=>fatal(err,cap,"density evaluator panic in workspace") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void) { let _=catch_unwind(AssertUnwindSafe(|| if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}})); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void, workspace:*mut c_void, q:*const f64, ndim:usize, lp:*mut f64, gradient:*mut f64, err:*mut c_char, cap:usize) -> i32 {
 let answer=catch_unwind(AssertUnwindSafe(|| -> Result<i32,String> { if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())}; let b=unsafe{&*bound.cast::<Bound>()}; if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())}; let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}}; let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)}; let value=eval_model(&b.data,q,g)?; if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}; unsafe{*lp=value}; Ok(0) }));
 match answer { Ok(Ok(status))=>status, Ok(Err(s))=>fatal(err,cap,s), Err(_)=>fatal(err,cap,"density evaluator panic in evaluate") }
}
