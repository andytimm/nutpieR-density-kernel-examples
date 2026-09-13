use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county: Vec<usize>, uppm: Vec<f64>, floor: Vec<f64>, radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn num_array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must contain finite numerics"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let root=Value::Object(o.clone());
    let j=integer(&root,"J")?; let n=integer(&root,"N")?;
    if j==0 { return Err("J must be positive".into()); }
    let countyv=root.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if countyv.len()!=n { return Err("county_idx length mismatch".into()); }
    let mut county=Vec::with_capacity(n);
    for x in countyv { let k=x.as_u64().ok_or("county_idx must contain integers")? as usize; if k==0 || k>j { return Err("county_idx out of bounds".into()); } county.push(k-1); }
    Ok(Data {j,n,county,uppm:num_array(&root,"log_uppm",n)?,floor:num_array(&root,"floor_measure",n)?,radon:num_array(&root,"log_radon",n)?})
}
fn expected_layout(d: &Data) -> String {
    let mut names=Vec::with_capacity(d.j+5);
    for i in 1..=d.j { names.push(format!("alpha.{i}")); }
    names.extend(["beta.1".into(),"beta.2".into(),"mu_alpha".into(),"sigma_alpha".into(),"sigma_y".into()]); names.join("\n")
}
fn eval_model(d: &Data,q:&[f64],g:&mut[f64])->Result<f64,String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0); let j=d.j; let b1=q[j]; let b2=q[j+1]; let ma=q[j+2]; let lsa=q[j+3]; let lsy=q[j+4];
    let sa=lsa.exp(); let sy=lsy.exp(); if !sa.is_finite() || !sy.is_finite() { return Err("transform overflow".into()); }
    let isa2=1.0/(sa*sa); let isy2=1.0/(sy*sy);
    // Propto=true keeps scale terms whose scale is a parameter. Positive transforms add log Jacobians.
    let mut lp = -(d.n as f64) * 0.5 * (2.0 * std::f64::consts::PI).ln() + lsa + lsy - 0.5*sa*sa - 0.5*sy*sy - 0.5*(ma/10.0).powi(2) - 0.5*(b1/10.0).powi(2) - 0.5*(b2/10.0).powi(2);
    g[j] = -b1/100.0; g[j+1] = -b2/100.0; g[j+2] = -ma/100.0;
    // sigma priors plus Jacobians
    g[j+3] = 1.0-sa*sa; g[j+4] = 1.0-sy*sy;
    for a in 0..j { let z=q[a]-ma; lp += -lsa-0.5*z*z*isa2; g[a] += -z*isa2; g[j+2] += z*isa2; g[j+3] += -1.0+z*z*isa2; }
    for n in 0..d.n { let a=d.county[n]; let r=d.radon[n]-(q[a]+d.uppm[n]*b1+d.floor[n]*b2); lp += -lsy-0.5*r*r*isy2; let w=r*isy2; g[a]+=w; g[j]+=w*d.uppm[n]; g[j+1]+=w*d.floor[n]; g[j+4]+=-1.0+r*r*isy2; }
    Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.j+5||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast();};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w;};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=value;};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
