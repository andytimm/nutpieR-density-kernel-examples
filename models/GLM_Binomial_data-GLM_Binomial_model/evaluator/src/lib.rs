use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable input copied at bind. year_squared remains an evaluation-local
// expression so the Stan transformed-data calculation is not replaced by a
// bind-time data-only summary.
struct Data { counts: Vec<f64>, totals: Vec<f64>, year: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonnegative_integer(v: &Value, key: &str) -> Result<usize, String> {
    v.as_u64().and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn array_numbers(v: &Value, key: &str, n: usize, integer: bool) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|x| {
        let z = if integer { x.as_u64().map(|q| q as f64) } else { x.as_f64() };
        z.filter(|q| q.is_finite()).ok_or_else(|| format!("{key} must contain finite {}", if integer { "integers" } else { "numbers" }))
    }).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = nonnegative_integer(o.get("nyears").ok_or("missing nyears")?, "nyears")?;
    let counts = array_numbers(&v, "C", n, true)?;
    let totals = array_numbers(&v, "N", n, true)?;
    let year = array_numbers(&v, "year", n, false)?;
    if counts.iter().zip(&totals).any(|(c, total)| c > total) { return Err("C must not exceed N".into()); }
    Ok(Data { counts, totals, year })
}
fn expected_layout(_d: &Data) -> String { "alpha\nbeta1\nbeta2".into() }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let (alpha, beta1, beta2) = (q[0], q[1], q[2]);
    // normal(0, 100), propto=TRUE: parameter-dependent quadratic only.
    let prior_scale_sq = 10_000.0;
    let mut lp = -0.5 * (alpha * alpha + beta1 * beta1 + beta2 * beta2) / prior_scale_sq;
    g[0] = -alpha / prior_scale_sq;
    g[1] = -beta1 / prior_scale_sq;
    g[2] = -beta2 / prior_scale_sq;
    for i in 0..d.year.len() {
        // This is exactly transformed data's year .* year, evaluated without
        // a bind-time reduction, then the binomial_logit log PMF without its
        // parameter-independent binomial coefficient.
        let y = d.year[i];
        let eta = alpha + beta1 * y + beta2 * (y * y);
        lp += d.counts[i] * eta - d.totals[i] * softplus(eta);
        let residual = d.counts[i] - d.totals[i] * sigmoid(eta);
        g[0] += residual;
        g[1] += residual * y;
        g[2] += residual * y * y;
    }
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
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?; let data=parse_data(v)?; let want=expected_layout(&data); let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?; if ndim!=3||got!=want{return Err("dimension/layout mismatch".into())}; Ok(Bound{data,ndim})})); match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
