use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, kid_score: Vec<f64>, mom_iq: Vec<f64>, mom_hs: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn numeric_array(o: &serde_json::Map<String, Value>, key: &str, n: usize, lo: f64, hi: f64) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite() && *x >= lo && *x <= hi)
        .ok_or_else(|| format!("{key} must contain finite values in bounds"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).filter(|&x| x <= usize::MAX as u64)
        .ok_or("N must be a nonnegative integer")? as usize;
    Ok(Data { n,
        kid_score: numeric_array(o, "kid_score", n, 0.0, 200.0)?,
        mom_iq: numeric_array(o, "mom_iq", n, 0.0, 200.0)?,
        mom_hs: numeric_array(o, "mom_hs", n, 0.0, 1.0)?,
    })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite unconstrained position".into()); }
    let sigma = q[3].exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform out of domain".into()); }
    let inv_sigma = 1.0 / sigma;
    let inv_sigma_sq = inv_sigma * inv_sigma;
    let mut lp = q[3]; // lower-bound transform Jacobian
    let mut gb0 = 0.0; let mut gb1 = 0.0; let mut gb2 = 0.0;
    let mut glog_sigma = 1.0 - d.n as f64;
    for i in 0..d.n {
        let mu = q[0] + q[1] * d.mom_hs[i] + q[2] * d.mom_iq[i];
        let residual = d.kid_score[i] - mu;
        let scaled = residual * inv_sigma;
        lp += -0.5 * scaled * scaled - q[3];
        let score = residual * inv_sigma_sq;
        gb0 += score;
        gb1 += score * d.mom_hs[i];
        gb2 += score * d.mom_iq[i];
        glog_sigma += scaled * scaled;
    }
    // cauchy(0, 2.5)_lupdf on the constrained sigma scale.
    let z = sigma / 2.5;
    lp -= (z * z).ln_1p();
    glog_sigma -= 2.0 * z * z / (1.0 + z * z);
    g[0] = gb0; g[1] = gb1; g[2] = gb2; g[3] = glog_sigma;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())};2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=4||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
