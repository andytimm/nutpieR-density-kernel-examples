use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Data are copied at bind.  age indicators reproduce the Stan transformed-data block.
struct Data { y: Vec<f64>, x: Vec<[f64; 8]>, age30_44: Vec<f64>, age45_64: Vec<f64>, age65up: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))
}
fn vector(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|v| number(v, key)).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    let y = vector(o, "partyid7", n)?;
    let real_ideo = vector(o, "real_ideo", n)?;
    let race_adj = vector(o, "race_adj", n)?;
    let educ1 = vector(o, "educ1", n)?;
    let gender = vector(o, "gender", n)?;
    let income = vector(o, "income", n)?;
    let ages = o.get("age_discrete").and_then(Value::as_array).ok_or("age_discrete must be an integer array")?;
    if ages.len() != n { return Err("age_discrete has wrong length".into()); }
    let mut x = Vec::with_capacity(n);
    let mut age30_44 = Vec::with_capacity(n);
    let mut age45_64 = Vec::with_capacity(n);
    let mut age65up = Vec::with_capacity(n);
    for i in 0..n {
        let a = ages[i].as_i64().ok_or("age_discrete must be an integer array")?;
        // Exact transformed-data work from model.stan.
        age30_44.push((a == 2) as u8 as f64);
        age45_64.push((a == 3) as u8 as f64);
        age65up.push((a == 4) as u8 as f64);
        x.push([real_ideo[i], race_adj[i], educ1[i], gender[i], income[i], 0.0, 0.0, 0.0]);
    }
    Ok(Data { y, x, age30_44, age45_64, age65up })
}
fn expected_layout() -> String { (1..=9).map(|i| format!("beta.{i}")).chain(std::iter::once("sigma".to_string())).collect::<Vec<_>>().join("\n") }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|v| !v.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let u = q[9];
    let inv_sigma = (-u).exp();
    if !inv_sigma.is_finite() { return Err("sigma transform overflow".into()); }
    let inv_var = inv_sigma * inv_sigma;
    let mut ss = 0.0;
    for i in 0..d.y.len() {
        let z = &d.x[i];
        let mu = q[0] + q[1]*z[0] + q[2]*z[1]
            + q[3]*d.age30_44[i] + q[4]*d.age45_64[i] + q[5]*d.age65up[i]
            + q[6]*z[2] + q[7]*z[3] + q[8]*z[4];
        let r = d.y[i] - mu;
        let w = r * inv_var;
        ss += r * w;
        g[0] += w; g[1] += w*z[0]; g[2] += w*z[1];
        g[3] += w*d.age30_44[i]; g[4] += w*d.age45_64[i]; g[5] += w*d.age65up[i];
        g[6] += w*z[2]; g[7] += w*z[3]; g[8] += w*z[4];
    }
    // normal_lpdf(... | mu, sigma) under propto=true, plus lower-bound Jacobian u.
    g[9] = ss - d.y.len() as f64 + 1.0;
    Ok(-0.5 * ss - (d.y.len() as f64 - 1.0) * u)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=10||got!=expected_layout(){return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
