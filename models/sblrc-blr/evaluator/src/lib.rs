use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

/// Immutable direct copies of the supplied Stan data. X is row-major; this is a
/// representation change only, not bind-time likelihood reduction.
struct Data { n: usize, d: usize, x: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, what: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{what} must be finite numeric"))
}
fn nonnegative_integer(v: &Value, what: &str) -> Result<usize, String> {
    let x = finite_num(v, what)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 { Err(format!("{what} must be a nonnegative integer")) }
    else { Ok(x as usize) }
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = nonnegative_integer(o.get("N").ok_or("missing N")?, "N")?;
    let d = nonnegative_integer(o.get("D").ok_or("missing D")?, "D")?;
    let ya = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if ya.len() != n { return Err("y length does not match N".into()); }
    let mut y = Vec::with_capacity(n);
    for (i, z) in ya.iter().enumerate() { y.push(finite_num(z, &format!("y[{i}]"))?); }
    let xa = o.get("X").and_then(Value::as_array).ok_or("X must be an array")?;
    if xa.len() != n { return Err("X rows do not match N".into()); }
    let mut x = Vec::with_capacity(n.checked_mul(d).ok_or("X size overflow")?);
    for (i, row) in xa.iter().enumerate() {
        let row = row.as_array().ok_or_else(|| format!("X row {i} must be an array"))?;
        if row.len() != d { return Err(format!("X row {i} length does not match D")); }
        for (j, z) in row.iter().enumerate() { x.push(finite_num(z, &format!("X[{i},{j}]"))?); }
    }
    Ok(Data { n, d, x, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.d).map(|j| format!("beta.{j}")).collect();
    names.push("sigma".into());
    names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let beta = &q[..d.d];
    let z = q[d.d];
    if !z.is_finite() { return Err("non-finite unconstrained sigma".into()); }
    let sigma = z.exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform outside finite domain".into()); }
    let sigma2 = sigma * sigma;
    let inv_sigma2 = 1.0 / sigma2;
    let normal_const = -0.5 * (d.n + d.d + 1) as f64 * (2.0 * std::f64::consts::PI).ln() - (d.d + 1) as f64 * 10.0_f64.ln();
    // Explicit Stan normal_lpdf calls retain their normalizing constants.
    let mut lp = normal_const + z - d.n as f64 * z - 0.5 * sigma2 / 100.0;
    let mut gz = 1.0 - sigma2 / 100.0 - d.n as f64;
    for j in 0..d.d { g[j] = -beta[j] / 100.0; lp -= 0.5 * beta[j] * beta[j] / 100.0; }
    for i in 0..d.n {
        let row = &d.x[i*d.d..(i+1)*d.d];
        let mut eta = 0.0;
        for j in 0..d.d { eta += row[j] * beta[j]; }
        let r = d.y[i] - eta;
        lp -= 0.5 * r * r * inv_sigma2;
        let scaled = r * inv_sigma2;
        for j in 0..d.d { g[j] += row[j] * scaled; }
        gz += r * r * inv_sigma2;
    }
    g[d.d] = gz;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.d+1 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
