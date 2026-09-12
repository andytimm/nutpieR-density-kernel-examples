use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { m: usize, t: usize, y: Vec<u8>, observed: Vec<bool> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let m = integer(&Value::Object(o.clone()), "M")?;
    let t = integer(&Value::Object(o.clone()), "T")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != m { return Err("y row count does not match M".into()); }
    let mut y = Vec::with_capacity(m.checked_mul(t).ok_or("data dimensions overflow")?);
    let mut observed = Vec::with_capacity(m);
    for row in rows {
        let row = row.as_array().ok_or("y rows must be arrays")?;
        if row.len() != t { return Err("y column count does not match T".into()); }
        let mut seen = false;
        for x in row {
            let z = x.as_u64().ok_or("y must contain integer 0 or 1")?;
            if z > 1 { return Err("y must contain integer 0 or 1".into()); }
            seen |= z == 1;
            y.push(z as u8);
        }
        observed.push(seen);
    }
    Ok(Data { m, t, y, observed })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.t + d.m + 2);
    names.push("omega".to_string());
    for j in 1..=d.t { names.push(format!("mean_p.{j}")); }
    names.push("sigma".to_string());
    for i in 1..=d.m { names.push(format!("eps_raw.{i}")); }
    names.join("\n")
}
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
#[inline] fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
#[inline] fn log_sigmoid(x: f64) -> f64 { -softplus(-x) }
#[inline] fn logsumexp(a: f64, b: f64) -> f64 { let hi=a.max(b); hi + ((a-hi).exp()+(b-hi).exp()).ln() }

fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let omega_q = q[0];
    let omega = sigmoid(omega_q);
    let sigma_q = q[1 + d.t];
    let sigma_unit = sigmoid(sigma_q);
    let sigma = 5.0 * sigma_unit;
    let mut lp = log_sigmoid(omega_q) + log_sigmoid(-omega_q);
    g.fill(0.0);
    g[0] = 1.0 - 2.0 * omega; // omega transform Jacobian
    for j in 0..d.t {
        let z = q[1 + j];
        let p = sigmoid(z);
        lp += log_sigmoid(z) + log_sigmoid(-z);
        g[1 + j] = 1.0 - 2.0 * p; // mean_p transform Jacobian
    }
    lp += 5.0f64.ln() + log_sigmoid(sigma_q) + log_sigmoid(-sigma_q);
    g[1 + d.t] = 1.0 - 2.0 * sigma_unit; // sigma transform Jacobian
    for i in 0..d.m {
        let raw = q[2 + d.t + i];
        lp += -0.5 * raw * raw;
        g[2 + d.t + i] = -raw;
        let mut l = 0.0;
        let mut h = 0.0;
        for j in 0..d.t {
            let eta = q[1 + j] + sigma * raw;
            let p = sigmoid(eta);
            let yy = d.y[i * d.t + j] as f64;
            l += yy * eta - softplus(eta);
            h += yy - p;
        }
        let likelihood_weight = if d.observed[i] {
            lp += log_sigmoid(omega_q) + l;
            g[0] += 1.0 - omega;
            1.0
        } else {
            let a = log_sigmoid(omega_q) + l;
            let b = log_sigmoid(-omega_q);
            let whole = logsumexp(a, b);
            let w = (a - whole).exp();
            lp += whole;
            g[0] += w - omega;
            w
        };
        let dh = likelihood_weight * h;
        for j in 0..d.t { g[1 + j] += likelihood_weight * ((d.y[i*d.t+j] as f64) - sigmoid(q[1+j] + sigma*raw)); }
        g[1 + d.t] += dh * raw * (5.0 * sigma_unit * (1.0 - sigma_unit));
        g[2 + d.t + i] += dh * sigma;
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.m+data.t+2 { return Err(format!("dimension mismatch: host {ndim}, data M={} T={}", data.m, data.t)); } if got != want { return Err(format!("layout mismatch: host {} bytes, expected {} bytes", got.len(), want.len())); } Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
