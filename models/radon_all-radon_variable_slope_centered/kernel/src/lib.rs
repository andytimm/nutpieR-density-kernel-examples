use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn num_array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().enumerate().map(|(i, x)| x.as_f64().filter(|z| z.is_finite())
        .ok_or_else(|| format!("{key}[{i}] must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let root = Value::Object(o.clone());
    let j = count(&root, "J")?;
    let n = count(&root, "N")?;
    if j == 0 { return Err("J must be positive for this model".into()); }
    let floor = num_array(&root, "floor_measure", n)?;
    let y = num_array(&root, "log_radon", n)?;
    let ca = root.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if ca.len() != n { return Err("county_idx length mismatch".into()); }
    let mut county = Vec::with_capacity(n);
    for (i, x) in ca.iter().enumerate() {
        let k = x.as_u64().and_then(|z| usize::try_from(z).ok())
            .filter(|z| *z >= 1 && *z <= j).ok_or_else(|| format!("county_idx[{i}] out of bounds"))?;
        county.push(k - 1);
    }
    Ok(Data { j, n, county, floor, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.j + 4);
    names.push("alpha".to_owned());
    names.extend((1..=d.j).map(|i| format!("beta.{i}")));
    names.extend(["mu_beta".to_owned(), "sigma_beta".to_owned(), "sigma_y".to_owned()]);
    names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let alpha = q[0];
    let beta = &q[1..1 + d.j];
    let mu_beta = q[1 + d.j];
    let log_sigma_beta = q[2 + d.j];
    let log_sigma_y = q[3 + d.j];
    let sigma_beta = log_sigma_beta.exp();
    let sigma_y = log_sigma_y.exp();
    if !sigma_beta.is_finite() || !sigma_y.is_finite() { return Err("scale overflow".into()); }
    let inv_sb2 = 1.0 / (sigma_beta * sigma_beta);
    let inv_sy2 = 1.0 / (sigma_y * sigma_y);
    let mut lp = -0.5 * (alpha / 10.0).powi(2) - 0.5 * (mu_beta / 10.0).powi(2)
        - 0.5 * sigma_beta * sigma_beta + log_sigma_beta
        - 0.5 * sigma_y * sigma_y + log_sigma_y;
    g[0] = -alpha / 100.0;
    g[1 + d.j] = -mu_beta / 100.0;
    let mut sum_beta_sq = 0.0;
    for j in 0..d.j {
        let diff = beta[j] - mu_beta;
        lp += -0.5 * diff * diff * inv_sb2 - log_sigma_beta;
        g[1 + j] += -diff * inv_sb2;
        g[1 + d.j] += diff * inv_sb2;
        sum_beta_sq += diff * diff;
    }
    let mut sum_res_sq = 0.0;
    for n in 0..d.n {
        let eta = alpha + d.floor[n] * beta[d.county[n]];
        let residual = d.y[n] - eta;
        // `target += normal_lpdf(...)` in the source is explicit, so Stan retains
        // the normalizing constant even under the overall propto convention.
        lp += -0.5 * residual * residual * inv_sy2 - log_sigma_y
            - 0.5 * (2.0 * std::f64::consts::PI).ln();
        let score = residual * inv_sy2;
        g[0] += score;
        g[1 + d.county[n]] += d.floor[n] * score;
        sum_res_sq += residual * residual;
    }
    // Chain rule for Stan's positive transforms, including each log Jacobian.
    g[2 + d.j] = sigma_beta * (-sigma_beta - (d.j as f64) / sigma_beta + sum_beta_sq / sigma_beta.powi(3)) + 1.0;
    g[3 + d.j] = sigma_y * (-sigma_y - (d.n as f64) / sigma_y + sum_res_sq / sigma_y.powi(3)) + 1.0;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } }
}
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
    if out.is_null() { return fatal(err,cap,"null bound output"); } unsafe { *out=std::ptr::null_mut(); }
    let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{ let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?; let data=parse_data(v)?; let want=expected_layout(&data); let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?; if ndim!=data.j+4 || got!=want{return Err("dimension/layout mismatch".into())}; Ok(Bound{data,ndim}) }));
    match answer { Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=match eval_model(&b.data,q,g){Ok(v)=>v,Err(_)=>return Ok(1)};if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
