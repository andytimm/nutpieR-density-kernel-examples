use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, x: Vec<f64>, y: Vec<f64>, pmualpha: f64, psalpha: f64, pmubeta: f64, psbeta: f64 }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(o: &serde_json::Map<String, Value>, key: &str) -> Result<f64, String> {
    o.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn finite_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>().filter(|a| a.iter().all(|z| z.is_finite())).ok_or_else(|| format!("{key} must be finite numeric array"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).filter(|&n| n <= usize::MAX as u64).ok_or("N must be nonnegative integer")? as usize;
    let x = finite_array(o, "x", n)?;
    let y = finite_array(o, "y", n)?;
    let pmualpha = finite_num(o, "pmualpha")?;
    let psalpha = finite_num(o, "psalpha")?;
    let pmubeta = finite_num(o, "pmubeta")?;
    let psbeta = finite_num(o, "psbeta")?;
    if psalpha <= 0.0 || psbeta <= 0.0 { return Err("prior standard deviations must be > 0".into()); }
    Ok(Data { n, x, y, pmualpha, psalpha, pmubeta, psbeta })
}
fn expected_layout(_: &Data) -> String { "alpha\nbeta\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let alpha = q[0]; let beta = q[1]; let log_sigma = q[2];
    let sigma = log_sigma.exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("sigma transform outside numerical domain".into()); }
    let inv_var = 1.0 / (sigma * sigma);
    let da = alpha - d.pmualpha; let db = beta - d.pmubeta;
    let mut ss = 0.0; let mut sum_res = 0.0; let mut sum_xres = 0.0;
    for i in 0..d.n {
        let r = d.y[i] - (alpha + beta * d.x[i]);
        ss += r * r; sum_res += r; sum_xres += d.x[i] * r;
    }
    let pa_var = d.psalpha * d.psalpha; let pb_var = d.psbeta * d.psbeta;
    // Stan normal_lpdf(...), propto=true: keep only terms depending on parameters.
    // sigma = exp(log_sigma), with its lower-bound log-Jacobian included last.
    g[0] = -da / pa_var + sum_res * inv_var;
    g[1] = -db / pb_var + sum_xres * inv_var;
    g[2] = -(d.n as f64) + ss * inv_var + 1.0;
    Ok(-0.5 * da * da / pa_var - 0.5 * db * db / pb_var - (d.n as f64) * log_sigma - 0.5 * ss * inv_var + log_sigma)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=3||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
