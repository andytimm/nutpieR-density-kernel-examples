use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Input values are copied at bind time. `y` is row-major, matching Stan's
// nested array indexing; no data-derived likelihood summaries are retained.
struct Data { m: usize, t: usize, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn as_usize(v: &Value, key: &str) -> Result<usize, String> {
    let n = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(n).map_err(|_| format!("{key} too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let m = as_usize(&v, "M")?;
    let t = as_usize(&v, "T")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != m { return Err("y row count must equal M".into()); }
    let mut y = Vec::with_capacity(m.checked_mul(t).ok_or("M*T overflow")?);
    for row in rows {
        let row = row.as_array().ok_or("y rows must be arrays")?;
        if row.len() != t { return Err("y column count must equal T".into()); }
        for x in row {
            match x.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("y must contain 0 or 1".into()) }
        }
    }
    Ok(Data { m, t, y })
}
fn expected_layout(_: &Data) -> String { "omega\np\nc".into() }
fn inv_logit(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let z=x.exp(); z/(1.0+z) } }
fn log1m(x: f64) -> f64 { (-x).ln_1p() }
fn log_sigmoid(x: f64) -> f64 { -if x >= 0.0 { (-x).exp().ln_1p() } else { -x + x.exp().ln_1p() } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.len() != 3 { return Err("evaluation dimension".into()); }
    let omega = inv_logit(q[0]); let p = inv_logit(q[1]); let c = inv_logit(q[2]);
    let mut lp = 0.0; let mut dw = 0.0; let mut dp = 0.0; let mut dc = 0.0;
    for i in 0..d.m {
        let row = &d.y[i*d.t..(i+1)*d.t];
        let seen = row.iter().any(|&z| z == 1);
        if seen {
            lp += omega.ln(); dw += 1.0 / omega;
            for j in 0..d.t {
                let prob = if j == 0 || row[j-1] == 0 { p } else { c };
                let deriv = if row[j] == 1 { 1.0/prob } else { -1.0/(1.0-prob) };
                if j == 0 || row[j-1] == 0 { dp += deriv; } else { dc += deriv; }
                lp += if row[j] == 1 { prob.ln() } else { log1m(prob) };
            }
        } else {
            // This exactly preserves the Stan marginalization for undetected rows.
            let a = omega.ln() + (d.t as f64) * log1m(p);
            let b = log1m(omega);
            let max = a.max(b); let lse = max + ((a-max).exp() + (b-max).exp()).ln();
            let present = (a-lse).exp();
            lp += lse;
            dw += present / omega - (1.0-present) / (1.0-omega);
            dp += present * (-(d.t as f64) / (1.0-p));
        }
    }
    // Stan's lower=0,upper=1 transform, followed by its Jacobian adjustment.
    let values = [omega, p, c]; let derivs = [dw, dp, dc];
    for k in 0..3 { lp += log_sigmoid(q[k]) + log_sigmoid(-q[k]); g[k] = derivs[k] * values[k] * (1.0-values[k]) + 1.0 - 2.0*values[k]; }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=3||got!=expected_layout(&data){return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
