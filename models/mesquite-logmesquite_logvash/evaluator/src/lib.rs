use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Data are copied unchanged at bind.  The transformed-data log operations are
// intentionally performed in eval_model, matching the original Stan program.
struct Data { n: usize, weight: Vec<f64>, diam1: Vec<f64>, diam2: Vec<f64>, canopy_height: Vec<f64>, total_height: Vec<f64>, group: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn array(v: &Value, key: &str, n: usize, positive: bool) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} length must equal N")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite() && (!positive || *z > 0.0)).ok_or_else(|| format!("{key} must contain {}finite numbers", if positive { "positive " } else { "" }))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).filter(|n| *n <= usize::MAX as u64).ok_or("N must be a nonnegative integer")? as usize;
    let root=Value::Object(o.clone());
    Ok(Data { n, weight: array(&root,"weight",n,true)?, diam1: array(&root,"diam1",n,true)?, diam2: array(&root,"diam2",n,true)?, canopy_height: array(&root,"canopy_height",n,true)?, total_height: array(&root,"total_height",n,true)?, group: array(&root,"group",n,false)? })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nbeta.5\nbeta.6\nsigma".into() }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let sigma = q[6].exp();
    if !sigma.is_finite() { return Err("sigma transform overflow".into()); }
    let inv_var = 1.0 / (sigma * sigma);
    let mut lp = q[6]; // lower-bound transform log Jacobian
    let mut gu = 1.0;
    g.fill(0.0);
    for i in 0..d.n {
        // Original transformed data, evaluated directly rather than reduced at bind.
        let y = d.weight[i].ln();
        let x1 = (d.diam1[i] * d.diam2[i] * d.canopy_height[i]).ln();
        let x2 = (d.diam1[i] * d.diam2[i]).ln();
        let x3 = (d.diam1[i] / d.diam2[i]).ln();
        let x4 = d.total_height[i].ln();
        let eta = q[0] + q[1]*x1 + q[2]*x2 + q[3]*x3 + q[4]*x4 + q[5]*d.group[i];
        let r = y - eta;
        let scaled = r * inv_var;
        lp += -q[6] - 0.5 * r * scaled; // normal_lpdf propto=true retains -log(sigma)
        g[0] += scaled; g[1] += scaled*x1; g[2] += scaled*x2; g[3] += scaled*x3; g[4] += scaled*x4; g[5] += scaled*d.group[i];
        gu += -1.0 + r * scaled;
    }
    g[6] = gu;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=7||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
