use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    let ys = o.get("y").and_then(Value::as_array).ok_or("y must be a numeric array")?;
    if ys.len() != n { return Err("y length must equal N".into()); }
    let mut y = Vec::with_capacity(n);
    for x in ys {
        y.push(x.as_f64().filter(|x| x.is_finite()).ok_or("y must contain finite numbers")?);
    }
    Ok(Data { y })
}
fn expected_layout(_d: &Data) -> String { "mu.1\nmu.2\nsigma.1\nsigma.2\ntheta".into() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn log_sigmoid(x: f64) -> f64 { if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() } }
fn log1m_sigmoid(x: f64) -> f64 { if x >= 0.0 { -x - (-x).exp().ln_1p() } else { -x.exp().ln_1p() } }
fn log_add_exp(a: f64, b: f64) -> f64 { let m=a.max(b); m + ((a-m).exp()+(b-m).exp()).ln() }

fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let mu1 = q[0]; let gap = q[1].exp(); let mu2 = mu1 + gap;
    let s1 = q[2].exp(); let s2 = q[3].exp(); let theta = sigmoid(q[4]);
    if !gap.is_finite() || !s1.is_finite() || !s2.is_finite() { return Err("transform overflow".into()); }
    // Explicit normal_lpdf calls retain their normalizing terms under this Stan source.
    const LOG_SQRT_2PI: f64 = 0.91893853320467274178032973640562;
    let mut lp = -0.5 * (mu1 * mu1 + mu2 * mu2) / 4.0;
    lp += -0.5 * (s1 * s1 + s2 * s2) / 4.0 + q[1] + q[2] + q[3];
    lp += 5.0 * log_sigmoid(q[4]) + 5.0 * log1m_sigmoid(q[4]);
    let mut dmu1 = -mu1 / 4.0;
    let mut dmu2 = -mu2 / 4.0;
    let mut ds1 = -s1 / 4.0;
    let mut ds2 = -s2 / 4.0;
    let mut dqtheta = 5.0 - 10.0 * theta;
    for &y in &d.y {
        let r1 = y - mu1; let r2 = y - mu2;
        let l1 = log_sigmoid(q[4]) - LOG_SQRT_2PI - q[2] - 0.5 * (r1 / s1).powi(2);
        let l2 = log1m_sigmoid(q[4]) - LOG_SQRT_2PI - q[3] - 0.5 * (r2 / s2).powi(2);
        let z = log_add_exp(l1, l2); lp += z;
        let w1 = (l1-z).exp(); let w2 = 1.0-w1;
        dmu1 += w1 * r1 / (s1*s1);
        dmu2 += w2 * r2 / (s2*s2);
        ds1 += w1 * (-1.0/s1 + r1*r1/(s1*s1*s1));
        ds2 += w2 * (-1.0/s2 + r2*r2/(s2*s2*s2));
        dqtheta += w1 - theta;
    }
    g[0] = dmu1 + dmu2;
    g[1] = gap * dmu2 + 1.0;
    g[2] = s1 * ds1 + 1.0;
    g[3] = s2 * ds2 + 1.0;
    g[4] = dqtheta;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?; let data=parse_data(v)?; let want=expected_layout(&data); let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?; if ndim != 5 || got != want{return Err("dimension/layout mismatch".into())}; Ok(Bound{data,ndim})})); match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
