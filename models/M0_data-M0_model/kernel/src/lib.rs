use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Input copies only. The Stan transformed-data row sums are deliberately
// recomputed in eval_model, as in the original program.
struct Data { m: usize, t: usize, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn positive_int(o: &serde_json::Map<String, Value>, key: &str, allow_zero: bool) -> Result<usize, String> {
    let x = o.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    let n = usize::try_from(x).map_err(|_| format!("{key} is too large"))?;
    if !allow_zero && n == 0 { return Err(format!("{key} must be positive")); }
    Ok(n)
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let m = positive_int(o, "M", true)?;
    let t = positive_int(o, "T", true)?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != m { return Err("y row count does not match M".into()); }
    let mut y = Vec::with_capacity(m.checked_mul(t).ok_or("y dimensions overflow")?);
    for row in rows {
        let a = row.as_array().ok_or("each y row must be an array")?;
        if a.len() != t { return Err("y column count does not match T".into()); }
        for x in a {
            match x.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("y entries must be 0 or 1".into()) }
        }
    }
    Ok(Data { m, t, y })
}
fn expected_layout(_: &Data) -> String { "omega\np".into() }
#[inline] fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
#[inline] fn inv_logit(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
#[inline] fn log_sum_exp(a: f64, b: f64) -> f64 { let z=a.max(b); z + ((a-z).exp() + (b-z).exp()).ln() }
#[inline] fn log_choose(n: usize, k: usize) -> f64 {
    let k = k.min(n-k);
    (1..=k).map(|j| ((n-k+j) as f64).ln() - (j as f64).ln()).sum()
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let omega = inv_logit(q[0]); let p = inv_logit(q[1]);
    let log_omega = -softplus(-q[0]); let log1m_omega = -softplus(q[0]);
    let log_p = -softplus(-q[1]); let log1m_p = -softplus(q[1]);
    // Jacobians for lower=0, upper=1 transforms, in BridgeStan order.
    let mut lp = log_omega + log1m_omega + log_p + log1m_p;
    let mut go = 1.0 - 2.0 * omega;
    let mut gp = 1.0 - 2.0 * p;
    for i in 0..d.m {
        let row = &d.y[i*d.t..(i+1)*d.t];
        let s: usize = row.iter().map(|&x| x as usize).sum(); // original transformed data s[i]
        if s > 0 {
            lp += log_omega + log_choose(d.t, s) + (s as f64) * log_p + ((d.t-s) as f64) * log1m_p;
            go += 1.0 - omega;
            gp += (s as f64) * (1.0-p) - ((d.t-s) as f64) * p;
        } else {
            let a = log_omega + (d.t as f64) * log1m_p;
            let b = log1m_omega;
            let z = log_sum_exp(a, b);
            let r = (a-z).exp(); // Pr(z_i=1 | never detected), on log scale stably
            lp += z;
            go += r - omega;
            gp -= (d.t as f64) * p * r;
        }
    }
    g[0] = go; g[1] = gp;
    Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char, cap:usize, s:impl AsRef<str>)->i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=2||got!=expected_layout(&data){return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
