use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county: Vec<usize>, uppm: Vec<f64>, floor: Vec<f64>, radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    v.get(key).and_then(Value::as_u64).and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn vector(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let x = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if x.len() != n { return Err(format!("{key} length mismatch")); }
    x.iter().map(|a| a.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let value = Value::Object(o.clone());
    let j = integer(&value, "J")?;
    let n = integer(&value, "N")?;
    if j == 0 { return Err("J must be positive".into()); }
    let county_raw = value.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if county_raw.len() != n { return Err("county_idx length mismatch".into()); }
    let mut county = Vec::with_capacity(n);
    for x in county_raw { let k=x.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or("county_idx must be integer")?; if k == 0 || k > j { return Err("county_idx out of bounds".into()); } county.push(k-1); }
    Ok(Data { j, n, county, uppm: vector(&value,"log_uppm",n)?, floor: vector(&value,"floor_measure",n)?, radon: vector(&value,"log_radon",n)? })
}
fn expected_layout(d: &Data) -> String {
    let mut x: Vec<String> = (1..=d.j).map(|i| format!("alpha_raw.{i}")).collect();
    x.extend(["beta.1", "beta.2", "mu_alpha", "sigma_alpha", "sigma_y"].iter().map(|s| (*s).into())); x.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let b1=q[d.j]; let b2=q[d.j+1]; let ma=q[d.j+2]; let za=q[d.j+3]; let zy=q[d.j+4];
    let sa=za.exp(); let sy=zy.exp();
    if !sa.is_finite() || !sy.is_finite() { return Err("scale overflow".into()); }
    let sy2=sy*sy;
    let mut lp = za + zy - 0.5*sa*sa - 0.5*sy2 - 0.5*(ma/10.0)*(ma/10.0) - 0.5*(b1/10.0)*(b1/10.0) - 0.5*(b2/10.0)*(b2/10.0) - (d.n as f64) * 0.91893853320467274178; // Explicit normal_lpdf likelihood retains its normalizing constant.
    let mut dza = 1.0 - sa*sa;
    let mut dzy = 1.0 - sy2;
    g[d.j] = -b1/100.0; g[d.j+1] = -b2/100.0; g[d.j+2] = -ma/100.0;
    for i in 0..d.j { lp -= 0.5*q[i]*q[i]; g[i] = -q[i]; }
    for n in 0..d.n {
        let i=d.county[n]; let pred=ma + sa*q[i] + d.uppm[n]*b1 + d.floor[n]*b2; let r=d.radon[n]-pred; let dr=r/sy2;
        lp += -0.5*r*dr - zy;
        g[i] += dr*sa; g[d.j] += dr*d.uppm[n]; g[d.j+1] += dr*d.floor[n]; g[d.j+2] += dr;
        dza += dr*sa*q[i]; dzy += r*dr - 1.0;
    }
    g[d.j+3]=dza; g[d.j+4]=dzy; Ok(lp)
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a [u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0 {let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.j+5||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
