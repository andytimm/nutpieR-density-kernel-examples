use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key).and_then(Value::as_u64).and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn numeric_array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let root = Value::Object(o.clone());
    let n = integer(&root, "N")?;
    let j = integer(&root, "J")?;
    if n == 0 || j == 0 { return Err("N and J must be positive".into()); }
    let county_raw = root.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if county_raw.len() != n { return Err("county_idx length mismatch".into()); }
    let mut county = Vec::with_capacity(n);
    for x in county_raw { let k=x.as_u64().and_then(|z| usize::try_from(z).ok()).ok_or("county_idx must be integer")?; if k == 0 || k > j { return Err("county_idx out of bounds".into()); } county.push(k-1); }
    Ok(Data { n, j, county, floor: numeric_array(&root,"floor_measure",n)?, y: numeric_array(&root,"log_radon",n)? })
}
fn expected_layout(d: &Data) -> String {
    let mut n: Vec<String> = (1..=d.j).map(|k| format!("alpha.{k}")).collect();
    n.push("beta".into()); n.push("sigma_y".into()); n.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let beta_i=d.j; let sigma_i=d.j+1; let sigma=q[sigma_i].exp();
    if !sigma.is_finite() || sigma <= 0.0 { return Err("sigma transform overflow".into()); }
    let inv_var=1.0/(sigma*sigma);
    let mut lp=q[sigma_i] - 0.5*sigma*sigma - 0.5*q[beta_i]*q[beta_i]/100.0;
    for k in 0..d.j { lp -= 0.5*q[k]*q[k]/100.0; g[k]=-q[k]/100.0; }
    g[beta_i]=-q[beta_i]/100.0;
    let mut sum_scaled_sq=0.0;
    for i in 0..d.n {
        let resid=d.y[i]-q[d.county[i]]-q[beta_i]*d.floor[i];
        let scaled=resid*inv_var;
        lp += -sigma.ln()-0.5*resid*scaled-0.5*(2.0*std::f64::consts::PI).ln();
        g[d.county[i]] += scaled;
        g[beta_i] += scaled*d.floor[i];
        sum_scaled_sq += resid*scaled;
    }
    g[sigma_i]=1.0-sigma*sigma-(d.n as f64)+sum_scaled_sq;
    Ok(lp)
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{
 if out.is_null(){return fatal(err,cap,"null bound output");} unsafe{*out=std::ptr::null_mut();}
 let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.j+2||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));
 match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())}let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
