use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, weight: Vec<f64>, diam1: Vec<f64>, diam2: Vec<f64>, canopy_height: Vec<f64>, total_height: Vec<f64>, density: Vec<f64>, group: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn positive_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>().filter(|x| x.iter().all(|v| v.is_finite() && *v > 0.0)).ok_or_else(|| format!("{key} must contain finite values > 0"))
}
fn finite_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(Value::as_f64).collect::<Option<Vec<_>>>().filter(|x| x.iter().all(|v| v.is_finite())).ok_or_else(|| format!("{key} must contain finite values"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_f = finite_num(&Value::Object(o.clone()), "N")?;
    if n_f < 0.0 || n_f.fract() != 0.0 || n_f > usize::MAX as f64 { return Err("N must be a nonnegative integer".into()); }
    let n = n_f as usize;
    Ok(Data { n, weight: positive_vec(o, "weight", n)?, diam1: positive_vec(o, "diam1", n)?, diam2: positive_vec(o, "diam2", n)?, canopy_height: positive_vec(o, "canopy_height", n)?, total_height: positive_vec(o, "total_height", n)?, density: positive_vec(o, "density", n)?, group: finite_vec(o, "group", n)? })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nbeta.5\nbeta.6\nbeta.7\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let sigma = q[7].exp();
    if !sigma.is_finite() { return Err("sigma transform is non-finite".into()); }
    let inv_sigma = 1.0 / sigma;
    let inv_sigma_sq = inv_sigma * inv_sigma;
    g.fill(0.0);
    let mut lp = q[7]; // Jacobian for sigma = exp(q[7]).
    for i in 0..d.n {
        // Reproduce transformed-data logarithms inside each evaluation; no bind-time summaries.
        let x1=d.diam1[i].ln(); let x2=d.diam2[i].ln(); let x3=d.canopy_height[i].ln();
        let x4=d.total_height[i].ln(); let x5=d.density[i].ln(); let y=d.weight[i].ln();
        let mu=q[0]+q[1]*x1+q[2]*x2+q[3]*x3+q[4]*x4+q[5]*x5+q[6]*d.group[i];
        let r=y-mu;
        // normal_lpdf with propto=true retains -log(sigma), then add transform Jacobian above.
        lp += -0.5 * r * r * inv_sigma_sq - q[7];
        let c=r*inv_sigma_sq;
        g[0]+=c; g[1]+=c*x1; g[2]+=c*x2; g[3]+=c*x3; g[4]+=c*x4; g[5]+=c*x5; g[6]+=c*d.group[i];
        g[7]+=r*r*inv_sigma_sq-1.0;
    }
    g[7] += 1.0; // derivative of the exp transform Jacobian.
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe{put_error(p,cap,s.as_ref())};2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=8||got!=expected_layout(&d){return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
