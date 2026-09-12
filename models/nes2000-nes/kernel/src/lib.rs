use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, y: Vec<f64>, x: Vec<[f64; 9]> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn num_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length must equal N")); }
    a.iter().map(|v| v.as_f64().filter(|x|x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn int_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<i64>, String> {
    let a=o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length must equal N")); }
    a.iter().map(|v| v.as_i64().ok_or_else(|| format!("{key} must contain integers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let n=o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    let y=num_vec(o,"partyid7",n)?;
    let real_ideo=num_vec(o,"real_ideo",n)?;
    let race_adj=num_vec(o,"race_adj",n)?;
    let educ1=num_vec(o,"educ1",n)?;
    let gender=num_vec(o,"gender",n)?;
    let income=num_vec(o,"income",n)?;
    let age=int_vec(o,"age_discrete",n)?;
    let mut x=Vec::with_capacity(n);
    for i in 0..n {
        // Exact original transformed-data indicators, copied into the fused row layout.
        x.push([1.0,real_ideo[i],race_adj[i],(age[i]==2) as u8 as f64,
          (age[i]==3) as u8 as f64,(age[i]==4) as u8 as f64,educ1[i],gender[i],income[i]]);
    }
    Ok(Data { n, y, x })
}
fn expected_layout(_: &Data) -> String {
    "beta.1\nbeta.2\nbeta.3\nbeta.4\nbeta.5\nbeta.6\nbeta.7\nbeta.8\nbeta.9\nsigma".into()
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|v| !v.is_finite()) { return Err("non-finite position".into()); }
    let sigma=q[9].exp();
    if !sigma.is_finite() || sigma==0.0 { return Err("sigma transform out of range".into()); }
    let inv_sigma=1.0/sigma;
    let mut ss=0.0;
    for j in 0..9 { g[j]=0.0; }
    for i in 0..d.n {
        let row=&d.x[i];
        let mut eta=0.0;
        for j in 0..9 { eta += q[j]*row[j]; }
        let r=(d.y[i]-eta)*inv_sigma;
        ss += r*r;
        let scale=r*inv_sigma;
        for j in 0..9 { g[j] += scale*row[j]; }
    }
    // normal_lpdf with propto=true retains parameter-dependent -log(sigma);
    // sigma=exp(q[9]) contributes its lower-bound transform Jacobian q[9].
    g[9]=ss - d.n as f64 + 1.0;
    Ok(-0.5*ss - (d.n as f64)*q[9] + q[9])
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a [u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{
 if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=10||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))};}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))};}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
