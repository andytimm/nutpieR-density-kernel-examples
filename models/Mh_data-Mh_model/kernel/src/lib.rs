use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copy of the supplied model data. `c` reproduces the Stan transformed-data count.
struct Data { m: usize, t: i32, y: Vec<i32>, c: usize }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let m = o.get("M").and_then(Value::as_u64).ok_or("M must be a nonnegative integer")? as usize;
    let t64 = o.get("T").and_then(Value::as_u64).ok_or("T must be a nonnegative integer")?;
    if t64 > i32::MAX as u64 { return Err("T is too large".into()); }
    let t = t64 as i32;
    let a = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if a.len() != m { return Err("y length must equal M".into()); }
    let mut y = Vec::with_capacity(m);
    let mut c = 0;
    for x in a {
        let z = x.as_i64().ok_or("y must contain integers")?;
        if z < 0 || z > t as i64 { return Err("y must be in 0:T".into()); }
        if z > 0 { c += 1; }
        y.push(z as i32);
    }
    Ok(Data { m, t, y, c })
}
fn expected_layout(d: &Data) -> String {
    let mut out = String::from("omega\nmean_p\nsigma");
    for i in 1..=d.m { out.push('\n'); out.push_str("eps_raw."); out.push_str(&i.to_string()); }
    out
}
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let z=x.exp(); z/(1.0+z) } }
#[inline] fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
#[inline] fn log_sigmoid(x: f64) -> f64 { -softplus(-x) }
#[inline] fn log_choose(n: i32, k: i32) -> f64 {
    // `target += binomial_logit_lpmf` retains this data-dependent coefficient.
    // Use C(n, min(k, n-k)); the old loop incorrectly started at n-k for k > n/2.
    let r = k.min(n - k);
    let mut out = 0.0;
    for j in 1..=r { out += ((n - r + j) as f64).ln() - (j as f64).ln(); }
    out
}

// Exact Stan propto=true, jacobian=true target. Binomial coefficients and constrained
// uniform normalizers are parameter-independent and omitted by propto.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let omega = sigmoid(q[0]);
    let mean_p = sigmoid(q[1]);
    let sigma_unit = sigmoid(q[2]);
    let sigma = 5.0 * sigma_unit;
    let intercept = q[1]; // logit(inv_logit(q[1])) exactly, avoiding cancellation
    let mut lp = log_sigmoid(q[0]) + log_sigmoid(-q[0])
        + log_sigmoid(q[1]) + log_sigmoid(-q[1])
        + 5.0f64.ln() + log_sigmoid(q[2]) + log_sigmoid(-q[2]);
    let mut d_omega = 0.0;
    let mut d_eta_total = 0.0;
    let mut d_sigma = 0.0;
    for i in 0..d.m {
        let e = q[3+i];
        let eta = intercept + sigma * e;
        let sp = softplus(eta);
        let p = sigmoid(eta);
        let (term, de) = if d.y[i] > 0 {
            ((omega.ln()) + log_choose(d.t, d.y[i]) + (d.y[i] as f64)*eta - (d.t as f64)*sp, (d.y[i] as f64) - (d.t as f64)*p)
        } else {
            let a = omega.ln() - (d.t as f64)*sp;
            let b = (-omega).ln_1p();
            let r = sigmoid(a-b);
            let mx = a.max(b);
            (mx + ((a-mx).exp() + (b-mx).exp()).ln(), -r*(d.t as f64)*p)
        };
        // `eps_raw ~ normal(0, 1)` is emitted as normal_lpdf<propto__>.
        // With propto=true it retains its parameter-dependent quadratic,
        // -0.5 * eps_raw[i]^2, while omitting the constant normalizer.
        lp += -0.5 * e * e + term;
        if d.y[i] > 0 { d_omega += 1.0 / omega; }
        else { // derivative of log_sum_exp wrt constrained omega
            let a = omega.ln() - (d.t as f64)*sp;
            let b = (-omega).ln_1p();
            let r = sigmoid(a-b);
            d_omega += r/omega - (1.0-r)/(1.0-omega);
        }
        d_eta_total += de;
        d_sigma += de * e;
        g[3+i] = -e + de * sigma;
    }
    // transform chain rules plus transform log-Jacobian derivatives
    g[0] = d_omega * omega * (1.0-omega) + 1.0 - 2.0*omega;
    g[1] = d_eta_total + 1.0 - 2.0*mean_p;
    g[2] = d_sigma * 5.0*sigma_unit*(1.0-sigma_unit) + 1.0 - 2.0*sigma_unit;
    // c is deliberately retained: it is the model's transformed-data calculation.
    let _ = d.c;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.m+3 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
