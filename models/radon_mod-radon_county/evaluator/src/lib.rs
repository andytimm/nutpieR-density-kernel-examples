use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county: Vec<usize>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x = num(v, key)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 { Err(format!("{key} must be a nonnegative integer")) } else { Ok(x as usize) }
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = count(&v, "N")?; let j = count(&v, "J")?;
    let county_v = o.get("county").and_then(Value::as_array).ok_or("county must be an array")?;
    let y_v = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if county_v.len() != n || y_v.len() != n { return Err("county and y must have length N".into()); }
    if j == 0 && n != 0 { return Err("J must be positive when N is positive".into()); }
    let mut county = Vec::with_capacity(n); let mut y = Vec::with_capacity(n);
    for (i, x) in county_v.iter().enumerate() {
        let c = x.as_u64().ok_or_else(|| format!("county[{i}] must be integer"))? as usize;
        if c == 0 || c > j { return Err(format!("county[{i}] outside 1..J")); }
        county.push(c - 1);
    }
    for (i, x) in y_v.iter().enumerate() {
        y.push(x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("y[{i}] must be finite numeric"))?);
    }
    Ok(Data { n, j, county, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.j).map(|i| format!("a.{i}")).collect();
    names.push("mu_a".into()); names.push("sigma_a".into()); names.push("sigma_y".into());
    names.join("\n")
}
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let mu_i=d.j; let sa_i=d.j+1; let sy_i=d.j+2;
    let sa_s=sigmoid(q[sa_i]); let sy_s=sigmoid(q[sy_i]);
    let sigma_a=100.0*sa_s; let sigma_y=100.0*sy_s;
    if sigma_a == 0.0 || sigma_y == 0.0 { return Err("scale transform underflow".into()); }
    let inv_sa2=1.0/(sigma_a*sigma_a); let inv_sy2=1.0/(sigma_y*sigma_y);
    let mu=q[mu_i];
    let mut lp=-0.5*mu*mu;
    let mut ss_a=0.0; let mut ss_y=0.0;
    let mut grad_mu=-mu;
    for k in 0..d.j { let z=q[k]-mu; ss_a += z*z; lp -= 0.5*z*z*inv_sa2; g[k] -= z*inv_sa2; grad_mu += z*inv_sa2; }
    lp -= (d.j as f64)*sigma_a.ln();
    for i in 0..d.n { let r=d.y[i]-q[d.county[i]]; ss_y += r*r; lp -= 0.5*r*r*inv_sy2; g[d.county[i]] += r*inv_sy2; }
    lp -= (d.n as f64)*sigma_y.ln();
    // Stan lower/upper transforms: sigma = 100 * inv_logit(q), with log Jacobian.
    lp += 100.0_f64.ln() + sa_s.ln() + (1.0-sa_s).ln();
    lp += 100.0_f64.ln() + sy_s.ln() + (1.0-sy_s).ln();
    g[mu_i]=grad_mu;
    g[sa_i]=(-((d.j as f64)/sigma_a) + ss_a/(sigma_a*sigma_a*sigma_a))*100.0*sa_s*(1.0-sa_s) + 1.0-2.0*sa_s;
    g[sy_i]=(-((d.n as f64)/sigma_y) + ss_y/(sigma_y*sigma_y*sigma_y))*100.0*sy_s*(1.0-sy_s) + 1.0-2.0*sy_s;
    Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())};2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.j+3 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
