use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies of the supplied Stan data. `obsyear` is retained and checked
// although the original transformed-parameter assignment makes the likelihood
// independent of it; no data-only reduction is performed at bind time.
struct Data {
    nobs: usize,
    nmis: usize,
    nyear: usize,
    nsite: usize,
    obs: Vec<f64>,
    obsyear: Vec<usize>,
    obssite: Vec<usize>,
    misyear: Vec<usize>,
    missite: Vec<usize>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn nonneg_int(o: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    let x = o.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn int_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<usize>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|v| v.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or_else(|| format!("{key} must contain nonnegative integers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let nobs = nonneg_int(o, "nobs")?;
    let nmis = nonneg_int(o, "nmis")?;
    let nyear = nonneg_int(o, "nyear")?;
    let nsite = nonneg_int(o, "nsite")?;
    let obs_u = int_vec(o, "obs", nobs)?;
    let obsyear = int_vec(o, "obsyear", nobs)?;
    let obssite = int_vec(o, "obssite", nobs)?;
    let misyear = int_vec(o, "misyear", nmis)?;
    let missite = int_vec(o, "missite", nmis)?;
    if obsyear.iter().any(|&x| x == 0 || x > nyear) || misyear.iter().any(|&x| x == 0 || x > nyear) {
        return Err("year index outside 1:nyear".into());
    }
    if obssite.iter().any(|&x| x == 0 || x > nsite) || missite.iter().any(|&x| x == 0 || x > nsite) {
        return Err("site index outside 1:nsite".into());
    }
    Ok(Data { nobs, nmis, nyear, nsite, obs: obs_u.into_iter().map(|x| x as f64).collect(), obsyear, obssite, misyear, missite })
}
fn expected_layout(d: &Data) -> String {
    let mut names: Vec<String> = (1..=d.nsite).map(|i| format!("alpha.{i}")).collect();
    names.push("mu_alpha".into());
    names.push("sd_alpha".into());
    names.join("\n")
}
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn log_sigmoid(x: f64) -> f64 { if x >= 0.0 { -((-x).exp()).ln_1p() } else { x - x.exp().ln_1p() } }
// Exact Stan model in unconstrained coordinates: alpha, mu_alpha, then
// lower=0 upper=5 sd_alpha. The original log_lambda repetition is represented
// directly by alpha[obssite] in the fused observation loop.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let ns = d.nsite;
    let mu = q[ns];
    let zsd = q[ns + 1];
    let s = sigmoid(zsd);
    let sd = 5.0 * s;
    if !mu.is_finite() || !sd.is_finite() || sd <= 0.0 { return Err("non-finite parameter".into()); }
    g.fill(0.0);
    let inv_sd = 1.0 / sd;
    let inv_var = inv_sd * inv_sd;
    let mut lp = log_sigmoid(zsd) + log_sigmoid(-zsd) + 5.0_f64.ln();
    let mut sum_sq = 0.0;
    let mut sum_centered = 0.0;
    for j in 0..ns {
        let r = q[j] - mu;
        if !r.is_finite() { return Err("non-finite parameter".into()); }
        sum_sq += r * r;
        sum_centered += r;
        lp += -0.5 * r * r * inv_var - sd.ln();
        g[j] += -r * inv_var;
    }
    // Directly reproduce every original Poisson-log term (no count aggregation).
    for i in 0..d.nobs {
        let j = d.obssite[i] - 1;
        let eta = q[j];
        let rate = eta.exp();
        if !rate.is_finite() { return Err("numerical likelihood rejection".into()); }
        lp += d.obs[i] * eta - rate;
        g[j] += d.obs[i] - rate;
    }
    lp += -0.5 * (mu / 10.0) * (mu / 10.0);
    g[ns] = sum_centered * inv_var - mu / 100.0;
    let constrained_sd_grad = sum_sq * inv_sd * inv_var - (ns as f64) * inv_sd;
    g[ns + 1] = constrained_sd_grad * sd * (1.0 - s) + 1.0 - 2.0 * s;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.nsite+2 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
