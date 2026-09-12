use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies retain the original inputs. The Stan transformed-data
// computations (dist / 100 and the interaction) remain in eval_model.
struct Data { n: usize, switched: Vec<u8>, dist: Vec<f64>, arsenic: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn array<'a>(o: &'a serde_json::Map<String, Value>, key: &str, n: usize) -> Result<&'a Vec<Value>, String> {
    let x = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if x.len() != n { return Err(format!("{key} has wrong length")); }
    Ok(x)
}
fn finite_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    array(o, key, n)?.iter().map(|v| v.as_f64().filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_u64 = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")?;
    let n = usize::try_from(n_u64).map_err(|_| "N is too large")?;
    let switched = array(o, "switched", n)?.iter().map(|v| match v.as_u64() {
        Some(0) => Ok(0), Some(1) => Ok(1), _ => Err("switched must contain 0 or 1".to_owned())
    }).collect::<Result<Vec<_>, _>>()?;
    Ok(Data { n, switched, dist: finite_array(o, "dist", n)?, arsenic: finite_array(o, "arsenic", n)? })
}
fn expected_layout(_d: &Data) -> String { "alpha\nbeta.1\nbeta.2\nbeta.3".into() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    g.fill(0.0);
    let (alpha, b0, b1, b2) = (q[0], q[1], q[2], q[3]);
    let mut lp = 0.0;
    for i in 0..d.n {
        // Exact transformed-data order: dist100, inter, then x row.
        let dist100 = d.dist[i] / 100.0;
        let inter = dist100 * d.arsenic[i];
        let eta = alpha + b0 * dist100 + b1 * d.arsenic[i] + b2 * inter;
        let y = d.switched[i] as f64;
        lp += y * eta - softplus(eta); // bernoulli_logit_lpmf, propto=true
        let r = y - sigmoid(eta);
        g[0] += r; g[1] += r * dist100; g[2] += r * d.arsenic[i]; g[3] += r * inter;
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } }
}
fn fatal(p:*mut c_char, cap:usize, s:impl AsRef<str>)->i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=4||got!=expected_layout(&d){return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
