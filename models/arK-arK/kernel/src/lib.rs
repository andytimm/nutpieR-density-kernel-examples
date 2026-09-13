use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { k: usize, t: usize, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let k = integer(&Value::Object(o.clone()), "K")?;
    let t = integer(&Value::Object(o.clone()), "T")?;
    if k == 0 || t <= k { return Err("require K > 0 and T > K".into()); }
    let a = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if a.len() != t { return Err("length(y) must equal T".into()); }
    let mut y = Vec::with_capacity(t);
    for x in a { y.push(x.as_f64().filter(|z| z.is_finite()).ok_or("y must be finite numeric")?); }
    Ok(Data { k, t, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.k + 2);
    names.push("alpha".to_string());
    for k in 1..=d.k { names.push(format!("beta.{k}")); }
    names.push("sigma".to_string());
    names.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    // BridgeStan propto=true, jacobian=true. q is alpha, beta[1:K], log(sigma).
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let alpha = q[0];
    let u = q[d.k + 1];
    let sigma = u.exp();
    if !sigma.is_finite() { return Ok(f64::NAN); }
    let inv_s2 = 1.0 / (sigma * sigma);
    let mut lp = -0.5 * alpha * alpha / 100.0;
    g.fill(0.0);
    g[0] = -alpha / 100.0;
    for k in 0..d.k {
        let beta = q[k + 1];
        lp -= 0.5 * beta * beta / 100.0;
        g[k + 1] = -beta / 100.0;
    }
    // cauchy(0, 2.5), retaining its parameter-dependent log1p term.
    let s2 = sigma * sigma;
    lp -= (s2 / 6.25).ln_1p();
    let mut grad_sigma = -2.0 * sigma / (6.25 + s2);
    for ti in d.k..d.t {
        let mut mu = alpha;
        for ki in 0..d.k { mu += q[ki + 1] * d.y[ti - ki - 1]; }
        let r = d.y[ti] - mu;
        lp += -u - 0.5 * r * r * inv_s2;
        let score = r * inv_s2;
        g[0] += score;
        for ki in 0..d.k { g[ki + 1] += score * d.y[ti - ki - 1]; }
        grad_sigma += -1.0 / sigma + r * r / (sigma * s2);
    }
    // sigma = exp(u): chain rule plus the log-Jacobian u.
    lp += u;
    g[d.k + 1] = grad_sigma * sigma + 1.0;
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
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let expected=data.k+2;if ndim!=expected||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
