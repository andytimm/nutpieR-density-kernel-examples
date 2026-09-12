use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n_dogs: usize, n_trials: usize, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let n = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(n).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n_dogs = integer(&v, "n_dogs")?;
    let n_trials = integer(&v, "n_trials")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != n_dogs { return Err("y row count mismatch".into()); }
    let mut y = Vec::with_capacity(n_dogs.checked_mul(n_trials).ok_or("data dimensions overflow")?);
    for row in rows {
        let values = row.as_array().ok_or("y rows must be arrays")?;
        if values.len() != n_trials { return Err("y column count mismatch".into()); }
        for value in values {
            match value.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("y values must be 0 or 1".into()) }
        }
    }
    Ok(Data { n_dogs, n_trials, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = vec!["mu_logit_ab.1".to_string(), "mu_logit_ab.2".to_string(),
        "sigma_logit_ab.1".to_string(), "sigma_logit_ab.2".to_string(), "L_logit_ab.1".to_string()];
    for col in 1..=2 { for row in 1..=d.n_dogs { names.push(format!("z.{row}.{col}")); } }
    names.join("\n")
}
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
#[inline] fn log_sigmoid(x: f64) -> f64 { -if x >= 0.0 { (-x).exp().ln_1p() } else { -x + x.exp().ln_1p() } }
#[inline] fn softplus(x: f64) -> f64 { if x >= 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() } }
// log(1-exp(x)) for x <= 0. Avoid cancellation close to zero.
#[inline] fn log1mexp(x: f64) -> f64 { if x > -std::f64::consts::LN_2 { (-x.exp_m1()).ln() } else { (-x.exp()).ln_1p() } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    g.fill(0.0);
    let mu1=q[0]; let mu2=q[1]; let s1=q[2].exp(); let s2=q[3].exp();
    let rho=q[4].tanh(); let l22=(1.0-rho*rho).sqrt();
    let mut lp = -mu1 - 2.0*softplus(-mu1) - mu2 - 2.0*softplus(-mu2);
    g[0] += 1.0 - 2.0*sigmoid(mu1); g[1] += 1.0 - 2.0*sigmoid(mu2);
    lp += -0.5*s1*s1 + q[2] -0.5*s2*s2 + q[3]; g[2] += 1.0-s1*s1; g[3] += 1.0-s2*s2;
    // For K=2, lkj_corr_cholesky(eta=1) contributes log(1-rho^2).
    // The Cholesky-correlation unconstraining Jacobian contributes it again.
    lp += 2.0 * (1.0-rho*rho).ln(); g[4] += -4.0*rho;
    for j in 0..d.n_dogs {
        let z1=q[5+j]; let z2=q[5+d.n_dogs+j];
        lp += -0.5*z1*z1 - 0.5*z2*z2; g[5+j] -= z1; g[5+d.n_dogs+j] -= z2;
        let x=mu1+z1*s1+z2*s2*rho; let w=mu2+z2*s2*l22;
        let a=sigmoid(x); let b=sigmoid(w);
        let mut shocks=0usize; let mut avoids=0usize;
        let mut dx=0.0; let mut dw=0.0;
        for t in 0..d.n_trials {
            let y=d.y[j*d.n_trials+t] as f64;
            let lprob=(shocks as f64)*log_sigmoid(x)+(avoids as f64)*log_sigmoid(w);
            // `lprob` is log(p), not logit(p): bernoulli(p) has a distinct
            // y=0 term and derivative.  See ../ledger.md.
            let coeff = if y == 1.0 { lp += lprob; 1.0 } else {
                let p = lprob.exp();
                lp += log1mexp(lprob);
                -p / (1.0 - p)
            };
            dx += coeff*(shocks as f64)*(1.0-a); dw += coeff*(avoids as f64)*(1.0-b);
            if y == 1.0 { shocks+=1; } else { avoids+=1; }
        }
        g[0]+=dx; g[1]+=dw; g[5+j]+=dx*s1; g[5+d.n_dogs+j]+=dx*s2*rho+dw*s2*l22;
        g[2]+=dx*z1*s1; g[3]+=dx*z2*s2*rho+dw*z2*s2*l22;
        let drho=dx*z2*s2-dw*z2*s2*rho/l22;
        g[4]+=drho*(1.0-rho*rho);
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe{put_error(p,cap,s.as_ref())};2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let expected=5+2*data.n_dogs;if ndim!=expected||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
