use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, y: Vec<f64>, sigma: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn num(v: &Value, name: &str) -> Result<f64, String> {
    v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{name} must be finite numeric"))
}
fn integer(v: &Value, name: &str) -> Result<usize, String> {
    let x = num(v, name)?;
    if x < 0. || x.fract() != 0. || x > usize::MAX as f64 { Err(format!("{name} must be a nonnegative integer")) }
    else { Ok(x as usize) }
}
fn array<'a>(o: &'a serde_json::Map<String, Value>, name: &str) -> Result<&'a Vec<Value>, String> {
    o.get(name).and_then(Value::as_array).ok_or_else(|| format!("{name} must be an array"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let j = integer(o.get("J").ok_or("missing J")?, "J")?;
    let ya = array(o, "y")?;
    let sa = array(o, "sigma")?;
    if ya.len() != j || sa.len() != j { return Err("y and sigma must have length J".into()); }
    let mut y = Vec::with_capacity(j);
    let mut sigma = Vec::with_capacity(j);
    for (i, x) in ya.iter().enumerate() { y.push(num(x, &format!("y[{}]", i + 1))?); }
    for (i, x) in sa.iter().enumerate() {
        let x = num(x, &format!("sigma[{}]", i + 1))?;
        if x < 0. { return Err(format!("sigma[{}] must be >= 0", i + 1)); }
        // A zero standard deviation is permitted by Stan's data declaration, but
        // normal_lpdf has an undefined scale there and therefore always rejects.
        if x == 0. { return Err(format!("sigma[{}] must be > 0 for a finite model density", i + 1)); }
        sigma.push(x);
    }
    Ok(Data { j, y, sigma })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.j).map(|i| format!("theta_trans.{i}")).collect();
    names.push("mu".into()); names.push("tau".into()); names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let j = d.j;
    let mu = q[j];
    let log_tau = q[j + 1];
    let tau = log_tau.exp();
    if !tau.is_finite() { return Err("tau transform is not finite".into()); }
    let mut lp = log_tau - 0.5 * (mu / 5.0).powi(2) - (1.0 + (tau / 5.0).powi(2)).ln();
    let mut g_mu = -mu / 25.0;
    let mut g_tau = -2.0 * tau / (25.0 + tau * tau);
    for i in 0..j {
        let z = q[i];
        let resid = (d.y[i] - (z * tau + mu)) / d.sigma[i];
        lp += -0.5 * z * z - 0.5 * resid * resid;
        g[i] = -z + resid * tau / d.sigma[i];
        g_mu += resid / d.sigma[i];
        g_tau += resid * z / d.sigma[i];
    }
    g[j] = g_mu;
    g[j + 1] = tau * g_tau + 1.0; // chain rule plus log-Jacobian of exp
    Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char, cap:usize, s:impl AsRef<str>)->i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim !=d.j+2||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
