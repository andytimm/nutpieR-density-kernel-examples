use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county_idx: Vec<usize>, floor: Vec<f64>, log_radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn count(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key).and_then(Value::as_u64).and_then(|x| usize::try_from(x).ok()).ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn numeric_vec(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must contain finite numeric values"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let obj=Value::Object(o.clone());
    let j=count(&obj,"J")?; let n=count(&obj,"N")?;
    if j==0 { return Err("J must be positive".into()); }
    let floor=numeric_vec(&obj,"floor_measure",n)?; let log_radon=numeric_vec(&obj,"log_radon",n)?;
    let county=v.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if county.len()!=n { return Err("county_idx length mismatch".into()); }
    let mut county_idx=Vec::with_capacity(n);
    for x in county { let k=x.as_u64().and_then(|z| usize::try_from(z).ok()).ok_or("county_idx must contain integers")?; if k==0 || k>j { return Err("county_idx out of bounds".into()); } county_idx.push(k-1); }
    Ok(Data {j,n,county_idx,floor,log_radon})
}
fn expected_layout(d: &Data) -> String {
    let mut v=Vec::with_capacity(d.j+4); v.push("alpha".to_string());
    for k in 1..=d.j { v.push(format!("beta_raw.{k}")); }
    v.push("mu_beta".into()); v.push("sigma_beta".into()); v.push("sigma_y".into()); v.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let alpha=q[0]; let mu=q[d.j+1]; let log_sb=q[d.j+2]; let log_sy=q[d.j+3];
    let sb=log_sb.exp(); let sy=log_sy.exp();
    if !sb.is_finite() || !sy.is_finite() { return Err("non-finite scale transform".into()); }
    let inv_sy2=1.0/(sy*sy);
    for x in g.iter_mut() {*x=0.0;}
    // The source uses explicit normal_lpdf for each observation; BridgeStan's
    // propto target retains this likelihood normalizer.
    let mut lp=-0.5*(alpha/10.0)*(alpha/10.0)-0.5*(mu/10.0)*(mu/10.0)
        - 0.5 * (2.0 * std::f64::consts::PI).ln() * d.n as f64;
    lp += -0.5*sb*sb + log_sb -0.5*sy*sy + log_sy;
    g[0]=-alpha/100.0; g[d.j+1]=-mu/100.0; g[d.j+2]=1.0-sb*sb; g[d.j+3]=1.0-sy*sy;
    for j in 0..d.j { let br=q[1+j]; lp += -0.5*br*br; g[1+j]=-br; }
    for i in 0..d.n {
        let j=d.county_idx[i]; let br=q[1+j]; let f=d.floor[i]; let eta=alpha+f*(mu+sb*br); let r=d.log_radon[i]-eta;
        lp += -0.5*r*r*inv_sy2-log_sy;
        let z=r*inv_sy2;
        g[0]+=z; g[1+j]+=f*sb*z; g[d.j+1]+=f*z; g[d.j+2]+=f*sb*br*z; g[d.j+3]+=r*r*inv_sy2-1.0;
    }
    Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let expected=data.j+4;if ndim!=expected||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast();};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w;};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=value;};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
