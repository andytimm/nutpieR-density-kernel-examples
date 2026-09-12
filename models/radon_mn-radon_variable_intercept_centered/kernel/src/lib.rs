use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonneg_int(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn num_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let wrap = Value::Object(o.clone());
    let j = nonneg_int(&wrap, "J")?;
    let n = nonneg_int(&wrap, "N")?;
    if j == 0 { return Err("J must be positive".into()); }
    let raw_county = o.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be integer array")?;
    if raw_county.len() != n { return Err("county_idx length mismatch".into()); }
    let mut county = Vec::with_capacity(n);
    for v in raw_county { let x = v.as_u64().ok_or("county_idx must be integer")?; if x == 0 || x as usize > j { return Err("county_idx out of bounds".into()); } county.push(x as usize - 1); }
    Ok(Data { j, n, county, floor: num_vec(o, "floor_measure", n)?, y: num_vec(o, "log_radon", n)? })
}
fn expected_layout(d: &Data) -> String {
    let mut x: Vec<String> = (1..=d.j).map(|i| format!("alpha.{i}")).collect();
    x.extend(["beta".into(), "mu_alpha".into(), "sigma_alpha".into(), "sigma_y".into()]); x.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let j=d.j; let beta=q[j]; let mu=q[j+1]; let log_sa=q[j+2]; let log_sy=q[j+3];
    let sa=log_sa.exp(); let sy=log_sy.exp(); if !sa.is_finite() || !sy.is_finite() { return Err("scale transform overflow".into()); }
    let isa2=1.0/(sa*sa); let isy2=1.0/(sy*sy);
    let mut lp = -0.5*sa*sa + log_sa -0.5*sy*sy + log_sy -0.5*(mu/10.0)*(mu/10.0) -0.5*(beta/10.0)*(beta/10.0) - (j as f64)*log_sa - (d.n as f64)*log_sy - 0.5*(d.n as f64)*(2.0*std::f64::consts::PI).ln();
    for k in 0..j { let z=q[k]-mu; lp -= 0.5*z*z*isa2; g[k] = -z*isa2; }
    let mut sum_z=0.0; let mut sum_z2=0.0; let mut sum_r2=0.0; let mut gb=-beta/100.0;
    for i in 0..d.n { let a=d.county[i]; let r=d.y[i]-(q[a]+d.floor[i]*beta); lp -= 0.5*r*r*isy2; g[a] += r*isy2; gb += d.floor[i]*r*isy2; sum_r2 += r*r; }
    for k in 0..j { let z=q[k]-mu; sum_z += z; sum_z2 += z*z; }
    g[j]=gb; g[j+1]=sum_z*isa2-mu/100.0; g[j+2]= -sa*sa + 1.0 + sum_z2*isa2 - j as f64; g[j+3]= -sy*sy + 1.0 + sum_r2*isy2 - d.n as f64;
    Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a [u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=d.j+4||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
