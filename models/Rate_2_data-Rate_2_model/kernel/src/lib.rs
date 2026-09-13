use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies of the four supplied integer data values. No summaries are made.
struct Data { n1: u64, n2: u64, k1: u64, k2: u64 }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonnegative_integer(v: &Value, key: &str) -> Result<u64, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    Ok(x)
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n1 = nonnegative_integer(&Value::Object(o.clone()), "n1")?;
    let n2 = nonnegative_integer(&Value::Object(o.clone()), "n2")?;
    let k1 = nonnegative_integer(&Value::Object(o.clone()), "k1")?;
    let k2 = nonnegative_integer(&Value::Object(o.clone()), "k2")?;
    if n1 < 1 || n2 < 1 { return Err("n1 and n2 must be at least 1".into()); }
    if k1 > n1 || k2 > n2 { return Err("counts must not exceed trials".into()); }
    Ok(Data { n1, n2, k1, k2 })
}
fn expected_layout(_: &Data) -> String { "theta1\ntheta2".into() }

fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e = x.exp(); e / (1.0 + e) }
}
fn log_inv_logit(x: f64) -> f64 { if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() } }
fn log1m_inv_logit(x: f64) -> f64 { if x >= 0.0 { -x - (-x).exp().ln_1p() } else { -x.exp().ln_1p() } }

// beta(1,1), binomial likelihood, and lower/upper transform Jacobians.
// `binomial_lpmf<propto__>` drops its data-only combinatorial normalizer.
fn eval_one(n: u64, k: u64, x: f64) -> (f64, f64) {
    let theta = inv_logit(x);
    let a = k as f64 + 1.0;
    let b = (n - k) as f64 + 1.0;
    (a * log_inv_logit(x) + b * log1m_inv_logit(x), a - (n as f64 + 2.0) * theta)
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let (lp1, g1) = eval_one(d.n1, d.k1, q[0]);
    let (lp2, g2) = eval_one(d.n2, d.k2, q[1]);
    g[0] = g1; g[1] = g2;
    Ok(lp1 + lp2)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char, cap:usize, s:impl AsRef<str>)->i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != 2 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})})); match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
