use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

const P: usize = 9;
struct Data { n: usize, y: Vec<f64>, x: Vec<[f64; 9]> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn num_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn int_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<i64>, String> {
    let a=o.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|v| v.as_i64().ok_or_else(||format!("{key} must be integer"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let n=o.get("N").and_then(Value::as_u64).ok_or("N must be nonnegative integer")? as usize;
    let y=num_array(o,"partyid7",n)?;
    let ideo=num_array(o,"real_ideo",n)?; let race=num_array(o,"race_adj",n)?;
    let educ=num_array(o,"educ1",n)?; let gender=num_array(o,"gender",n)?; let income=num_array(o,"income",n)?;
    let age=int_array(o,"age_discrete",n)?;
    let mut x=Vec::with_capacity(n);
    for i in 0..n {
        if !(1..=4).contains(&age[i]) { return Err("age_discrete must be in 1..4".into()); }
        // Exact transformed-data indicators, represented only as repacked input.
        x.push([1.0, ideo[i], race[i], (age[i]==2) as i32 as f64, (age[i]==3) as i32 as f64,
                (age[i]==4) as i32 as f64, educ[i], gender[i], income[i]]);
    }
    Ok(Data {n,y,x})
}
fn expected_layout(_: &Data) -> String {
    (1..=9).map(|i| format!("beta.{i}")).chain(std::iter::once("sigma".to_string())).collect::<Vec<_>>().join("\n")
}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String> {
    if !q.iter().all(|x|x.is_finite()) { return Err("non-finite position".into()); }
    let log_sigma=q[P];
    // Stan lower=0 transform: sigma=exp(q), and its Jacobian is +q.
    let sigma=log_sigma.exp();
    if !sigma.is_finite() || sigma==0.0 { return Err("sigma transform outside finite domain".into()); }
    let inv_sigma=1.0/sigma; let inv_var=inv_sigma*inv_sigma;
    g.fill(0.0); let mut ss=0.0;
    for i in 0..d.n {
        let eta: f64=(0..P).map(|j|q[j]*d.x[i][j]).sum();
        let r=d.y[i]-eta;
        ss += r*r;
        let factor=r*inv_var;
        for j in 0..P { g[j] += factor*d.x[i][j]; }
    }
    // normal_lpdf propto: -0.5 sum squared residuals - N log(sigma);
    // add lower-bound transform log-Jacobian +log(sigma).
    g[P]=ss*inv_var - d.n as f64 + 1.0;
    Ok(-0.5*ss*inv_var - (d.n as f64)*log_sigma + log_sigma)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{
 if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=10||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
