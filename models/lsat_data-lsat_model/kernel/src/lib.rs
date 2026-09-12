use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies reproduce the Stan transformed-data expansion r[k, j].
struct Data { n: usize, t: usize, r: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn int_field(o: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    let x = o.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = int_field(o, "N")?;
    let r_count = int_field(o, "R")?;
    let t = int_field(o, "T")?;
    if r_count == 0 || t == 0 { return Err("R and T must be positive for this Stan program".into()); }
    let culm_v = o.get("culm").and_then(Value::as_array).ok_or("culm must be an array")?;
    if culm_v.len() != r_count { return Err("culm length does not match R".into()); }
    let response_v = o.get("response").and_then(Value::as_array).ok_or("response must be an array")?;
    if response_v.len() != r_count { return Err("response rows do not match R".into()); }
    let mut culm = Vec::with_capacity(r_count);
    let mut prior = 0usize;
    for x in culm_v {
        let c = x.as_u64().and_then(|z| usize::try_from(z).ok()).ok_or("culm must contain nonnegative integers")?;
        if c < prior || c > n { return Err("culm must be nondecreasing and at most N".into()); }
        prior = c; culm.push(c);
    }
    if culm[r_count - 1] != n { return Err("culm[R] must equal N".into()); }
    let mut patterns = Vec::with_capacity(r_count * t);
    for row in response_v {
        let a = row.as_array().ok_or("response must be a rectangular array")?;
        if a.len() != t { return Err("response columns do not match T".into()); }
        for x in a {
            let y = x.as_u64().ok_or("response must contain integers")?;
            if y > 1 { return Err("response values must be 0 or 1".into()); }
            patterns.push(y as u8);
        }
    }
    // This is exactly the original transformed-data assignment, retained rather
    // than replacing repeated observations with counts or sufficient statistics.
    let mut expanded = Vec::with_capacity(n * t);
    let mut start = 0usize;
    for p in 0..r_count {
        for _j in start..culm[p] {
            expanded.extend_from_slice(&patterns[p*t..(p+1)*t]);
        }
        start = culm[p];
    }
    Ok(Data { n, t, r: expanded })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.n + d.t + 1);
    for k in 1..=d.t { names.push(format!("alpha.{k}")); }
    for i in 1..=d.n { names.push(format!("theta.{i}")); }
    names.push("beta".into()); names.join("\n")
}
#[inline] fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let beta_q = q[d.t + d.n];
    let beta = beta_q.exp();
    if !beta.is_finite() { return Err("beta transform overflow".into()); }
    g.fill(0.0);
    let mut lp = beta_q - 0.5 * (beta / 100.0).powi(2);
    g[d.t + d.n] = 1.0 - (beta * beta) / 10000.0;
    for k in 0..d.t {
        let a = q[k]; lp -= 0.5 * (a / 100.0).powi(2); g[k] -= a / 10000.0;
    }
    for i in 0..d.n {
        let theta_index = d.t + i;
        let theta = q[theta_index];
        lp -= 0.5 * theta * theta; g[theta_index] -= theta;
        for k in 0..d.t {
            let eta = beta * theta - q[k];
            let y = d.r[i*d.t + k] as f64;
            lp += y * eta - softplus(eta);
            let residual = y - sigmoid(eta);
            g[k] -= residual;
            g[theta_index] += beta * residual;
            g[d.t + d.n] += beta * theta * residual;
        }
    }
    Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let want_ndim=data.t+data.n+1;if ndim!=want_ndim||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};if q.iter().any(|x|!x.is_finite()){return Ok(1)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
