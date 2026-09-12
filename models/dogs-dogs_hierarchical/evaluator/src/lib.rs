use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies of supplied data and the original Stan transformed-data
// recurrence. `shock` and `avoid` are one exponent pair per likelihood term,
// not reduced outcome totals.
struct Data {
    n_dogs: usize,
    n_trials: usize,
    shock: Vec<u32>,
    avoid: Vec<u32>,
    y: Vec<u8>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonnegative_int(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key).and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_dogs = nonnegative_int(&v, "n_dogs")?;
    let n_trials = nonnegative_int(&v, "n_trials")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != n_dogs { return Err("y row count mismatch".into()); }
    let n = n_dogs.checked_mul(n_trials).ok_or("data dimensions overflow")?;
    let mut y = Vec::with_capacity(n);
    let mut shock = Vec::with_capacity(n);
    let mut avoid = Vec::with_capacity(n);
    for (j, row) in rows.iter().enumerate() {
        let xs = row.as_array().ok_or_else(|| format!("y row {j} must be an array"))?;
        if xs.len() != n_trials { return Err(format!("y column count mismatch in row {j}")); }
        // Exact original transformed-data recurrence, materialized once at bind.
        let (mut s, mut a) = (0u32, 0u32);
        for (t, x) in xs.iter().enumerate() {
            let yy = x.as_u64().ok_or_else(|| format!("y[{j},{t}] must be integer"))?;
            if yy > 1 { return Err(format!("y[{j},{t}] must be 0 or 1")); }
            shock.push(s); avoid.push(a); y.push(yy as u8);
            if yy == 1 { s = s.checked_add(1).ok_or("shock count overflow")?; }
            else { a = a.checked_add(1).ok_or("avoid count overflow")?; }
        }
    }
    Ok(Data { n_dogs, n_trials, shock, avoid, y })
}
fn expected_layout(_: &Data) -> String { "a\nb".into() }

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) }
    else { let z = x.exp(); z / (1.0 + z) }
}
#[inline]
fn log_sigmoid(x: f64) -> f64 { -if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() } }

fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.len() != 2 || !q[0].is_finite() || !q[1].is_finite() { return Err("non-finite position".into()); }
    let a = sigmoid(q[0]);
    let b = sigmoid(q[1]);
    // The two bounded-real transforms, with their log-Jacobians, are ordered a then b.
    let mut lp = log_sigmoid(q[0]) + log_sigmoid(-q[0]) + log_sigmoid(q[1]) + log_sigmoid(-q[1]);
    let (mut ga, mut gb) = (1.0 - 2.0 * a, 1.0 - 2.0 * b);
    for i in 0..d.y.len() {
        let s = d.shock[i] as f64;
        let v = d.avoid[i] as f64;
        // This is p = a ^ prev_shock * b ^ prev_avoid from the Stan source.
        let eta = s * a.ln() + v * b.ln();
        if d.y[i] == 1 {
            lp += eta;
            ga += s * (1.0 - a);
            gb += v * (1.0 - b);
        } else {
            let p = eta.exp();
            lp += (-p).ln_1p();
            let de = -p / (1.0 - p);
            ga += de * s * (1.0 - a);
            gb += de * v * (1.0 - b);
        }
    }
    g[0] = ga; g[1] = gb;
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

#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
    if out.is_null() { return fatal(err,cap,"null bound output"); }; unsafe { *out=std::ptr::null_mut(); }
    let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{
        let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;
        let data=parse_data(v)?; let want=expected_layout(&data);
        let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;
        if ndim != 2 || got != want { return Err("dimension/layout mismatch".into()); }; Ok(Bound{data,ndim})
    }));
    match answer { Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0}, Ok(Err(s))=>fatal(err,cap,s), Err(_)=>fatal(err,cap,"density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void) { let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}})); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
    if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())})); match answer {Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {
    let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}
}
