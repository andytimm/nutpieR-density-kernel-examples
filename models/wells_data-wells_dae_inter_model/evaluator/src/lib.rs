use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data {
    n: usize,
    switched: Vec<u8>,
    dist: Vec<f64>,
    arsenic: Vec<f64>,
    educ: Vec<f64>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be nonnegative integer")? as usize;
    if n > 10_000_000 { return Err("N is too large".into()); }
    let switched_v = o.get("switched").and_then(Value::as_array).ok_or("switched must be array")?;
    if switched_v.len() != n { return Err("switched length mismatch".into()); }
    let mut switched = Vec::with_capacity(n);
    for v in switched_v {
        match v.as_u64() { Some(0) => switched.push(0), Some(1) => switched.push(1), _ => return Err("switched must contain 0 or 1".into()) }
    }
    Ok(Data { n, switched, dist: finite_vec(o, "dist", n)?, arsenic: finite_vec(o, "arsenic", n)?, educ: finite_vec(o, "educ", n)? })
}
fn expected_layout(_: &Data) -> String { "alpha\nbeta.1\nbeta.2\nbeta.3\nbeta.4\nbeta.5\nbeta.6".into() }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    // This repeats the Stan transformed-data centering and interactions per evaluation.
    let n = d.n as f64;
    if n == 0.0 { return Err("N must be positive for Stan means".into()); }
    let (mut sum_dist, mut sum_arsenic, mut sum_educ) = (0.0, 0.0, 0.0);
    for i in 0..d.n { sum_dist += d.dist[i]; sum_arsenic += d.arsenic[i]; sum_educ += d.educ[i]; }
    let (mean_dist, mean_arsenic, mean_educ) = (sum_dist/n, sum_arsenic/n, sum_educ/n);
    g.fill(0.0);
    let (alpha, b) = (q[0], &q[1..7]);
    let mut lp = 0.0;
    for i in 0..d.n {
        let cd = (d.dist[i] - mean_dist) / 100.0;
        let ca = d.arsenic[i] - mean_arsenic;
        let ce = (d.educ[i] - mean_educ) / 4.0;
        // Evaluation-local named features replace only the fixed local array
        // and its loops. The transformed-data mean passes, feature order, and
        // row-wise likelihood spelling remain unchanged.
        let da = cd * ca;
        let de = cd * ce;
        let ae = ca * ce;
        let eta = alpha + b[0] * cd + b[1] * ca + b[2] * ce
            + b[3] * da + b[4] * de + b[5] * ae;
        lp += (d.switched[i] as f64) * eta - softplus(eta);
        let r = d.switched[i] as f64 - sigmoid(eta);
        g[0] += r;
        g[1] += r * cd;
        g[2] += r * ca;
        g[3] += r * ce;
        g[4] += r * da;
        g[5] += r * de;
        g[6] += r * ae;
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let data=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?; let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?; if ndim !=7 || got != expected_layout(&data){return Err("dimension/layout mismatch".into())}; Ok(Bound{data,ndim})})); match a {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim != b.ndim || q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};if q.iter().any(|x|!x.is_finite()){return Ok(1)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
