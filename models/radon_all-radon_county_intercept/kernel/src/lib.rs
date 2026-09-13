use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county_idx: Vec<usize>, floor_measure: Vec<f64>, log_radon: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} too large"))
}
fn finite_array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let root = Value::Object(o.clone());
    let n = count(&root, "N")?; let j = count(&root, "J")?;
    if j == 0 { return Err("J must be positive".into()); }
    let county = root.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if county.len() != n { return Err("county_idx length mismatch".into()); }
    let county_idx: Result<Vec<_>, _> = county.iter().map(|x| {
        let k = x.as_u64().ok_or_else(|| "county_idx must contain integers".to_string())?;
        let k = usize::try_from(k).map_err(|_| "county_idx too large".to_string())?;
        if k == 0 || k > j { Err::<usize, String>("county_idx out of bounds".into()) } else { Ok(k - 1) }
    }).collect();
    Ok(Data { n, j, county_idx: county_idx?, floor_measure: finite_array(&root, "floor_measure", n)?, log_radon: finite_array(&root, "log_radon", n)? })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.j).map(|i| format!("alpha.{i}")).collect();
    names.push("beta".into()); names.push("sigma_y".into()); names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    // q: alpha[1:J], beta, log(sigma_y). Exact Stan propto=true,jacobian=true.
    g.fill(0.0);
    let beta_i = d.j; let u_i = d.j + 1;
    let beta = q[beta_i]; let u = q[u_i];
    let sigma = u.exp();
    if !sigma.is_finite() || sigma == 0.0 { return Err("non-finite sigma transform".into()); }
    let inv_sigma = 1.0 / sigma; let inv_sigma_sq = inv_sigma * inv_sigma;
    let mut lp = u - 0.5 * sigma * sigma - 0.5 * beta * beta / 100.0;
    g[u_i] = 1.0 - sigma * sigma;
    g[beta_i] = -beta / 100.0;
    for a in 0..d.j { lp -= 0.5 * q[a] * q[a] / 100.0; g[a] = -q[a] / 100.0; }
    for n in 0..d.n {
        let a = d.county_idx[n]; let r = d.log_radon[n] - q[a] - beta * d.floor_measure[n];
        lp += -0.9189385332046727 - u - 0.5 * r * r * inv_sigma_sq;
        let dr = r * inv_sigma_sq;
        g[a] += dr; g[beta_i] += dr * d.floor_measure[n]; g[u_i] += -1.0 + r * r * inv_sigma_sq;
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.j+2 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
