use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

const I: usize = 21;
struct Data { n: Vec<f64>, N: Vec<f64>, x1: Vec<f64>, x2: Vec<f64>, x1x2: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn usize_value(v: &Value, key: &str) -> Result<usize, String> {
    let x = number(v, key)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 { Err(format!("{key} must be a nonnegative integer")) } else { Ok(x as usize) }
}
fn array(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != I { return Err(format!("{key} must have length {I}")); }
    a.iter().enumerate().map(|(i, x)| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key}[{}] must be finite numeric", i + 1))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let whole = Value::Object(o.clone());
    if usize_value(&whole, "I")? != I { return Err(format!("I must equal {I}")); }
    let n = array(&whole, "n")?; let N = array(&whole, "N")?;
    let x1 = array(&whole, "x1")?; let x2 = array(&whole, "x2")?;
    for j in 0..I { if n[j] < 0.0 || N[j] < 0.0 || n[j].fract() != 0.0 || N[j].fract() != 0.0 || n[j] > N[j] { return Err(format!("n/N invalid at {}", j + 1)); } }
    // This is the original Stan transformed-data statement, not an added summary.
    let x1x2 = x1.iter().zip(&x2).map(|(a,b)| a*b).collect();
    Ok(Data { n, N, x1, x2, x1x2 })
}
fn expected_layout() -> String {
    let mut n = vec!["alpha0".to_string(), "alpha1".to_string(), "alpha12".to_string(), "alpha2".to_string(), "tau".to_string()];
    n.extend((1..=I).map(|j| format!("b.{j}"))); n.join("\n")
}
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let z=x.exp(); z/(1.0+z) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let (a0,a1,a12,a2,u)=(q[0],q[1],q[2],q[3],q[4]);
    let tau=u.exp();
    let mut lp = -0.5e-6*(a0*a0+a1*a1+a12*a12+a2*a2) + (1e-3-1.0)*u - 1e-3*tau + u;
    g.fill(0.0); g[0]=-a0*1e-6; g[1]=-a1*1e-6; g[2]=-a12*1e-6; g[3]=-a2*1e-6;
    let mut gu=(1e-3-1.0) - 1e-3*tau + 1.0;
    for j in 0..I {
        let b=q[5+j];
        lp += 0.5*u - 0.5*tau*b*b;
        gu += 0.5 - 0.5*tau*b*b;
        let eta=a0+a1*d.x1[j]+a2*d.x2[j]+a12*d.x1x2[j]+b;
        lp += d.n[j]*eta - d.N[j]*softplus(eta);
        let r=d.n[j]-d.N[j]*sigmoid(eta);
        g[0]+=r; g[1]+=r*d.x1[j]; g[2]+=r*d.x1x2[j]; g[3]+=r*d.x2[j]; g[5+j]=r-tau*b;
    }
    g[4]=gu; Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=I+5||got!=expected_layout(){return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
