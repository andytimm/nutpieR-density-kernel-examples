use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county: Vec<usize>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = integer(&v, "N")?; let j = integer(&v, "J")?;
    if j == 0 { return Err("J must be positive".into()); }
    let ci = array(&v, "county_idx")?;
    let yy = array(&v, "log_radon")?;
    if ci.len() != n || yy.len() != n { return Err("data array length does not match N".into()); }
    let mut county = Vec::with_capacity(n); let mut y = Vec::with_capacity(n);
    for (i, (c, obs)) in ci.iter().zip(yy.iter()).enumerate() {
        let c = c.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or_else(|| format!("county_idx[{i}] must be integer"))?;
        if c == 0 || c > j { return Err(format!("county_idx[{i}] out of bounds")); }
        let obs = obs.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("log_radon[{i}] must be finite numeric"))?;
        county.push(c - 1); y.push(obs);
    }
    let _ = o; Ok(Data { n, j, county, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.j).map(|i| format!("alpha.{i}")).collect();
    names.extend(["mu_alpha".to_string(), "sigma_alpha".to_string(), "sigma_y".to_string()]);
    names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let mu_i = d.j; let usa_i = d.j + 1; let usy_i = d.j + 2;
    let mu = q[mu_i]; let usa = q[usa_i]; let usy = q[usy_i];
    let sa = usa.exp(); let sy = usy.exp();
    if !sa.is_finite() || !sy.is_finite() { return Err("scale transform overflow".into()); }
    let inv_sa2 = 1.0 / (sa * sa); let inv_sy2 = 1.0 / (sy * sy);
    let mut lp = -0.5 * sa * sa + usa - 0.5 * sy * sy + usy - 0.5 * (mu / 10.0) * (mu / 10.0);
    let mut grad_mu = -mu / 100.0;
    let mut grad_usa = 1.0 - sa * sa;
    let mut grad_usy = 1.0 - sy * sy;
    for a in 0..d.j {
        let r = q[a] - mu; let z2 = r * r * inv_sa2;
        lp += -0.5 * z2 - usa;
        g[a] += -r * inv_sa2;
        grad_mu += r * inv_sa2;
        grad_usa += z2 - 1.0;
    }
    // This is an explicit `target += normal_lpdf(...)` in the Stan source,
    // so its parameter-independent normalizing constant is retained under propto.
    lp -= 0.5 * (d.n as f64) * (2.0 * std::f64::consts::PI).ln();
    for i in 0..d.n {
        let a = d.county[i]; let r = d.y[i] - q[a]; let z2 = r * r * inv_sy2;
        lp += -0.5 * z2 - usy;
        g[a] += r * inv_sy2;
        grad_usy += z2 - 1.0;
    }
    g[mu_i] = grad_mu; g[usa_i] = grad_usa; g[usy_i] = grad_usy;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.j+3||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
