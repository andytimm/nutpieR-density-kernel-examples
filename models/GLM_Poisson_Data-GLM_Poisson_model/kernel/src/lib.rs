use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable binding copies input vectors. y2/y3 reproduce Stan's transformed-data work.
struct Data { counts: Vec<f64>, year: Vec<f64>, year_squared: Vec<f64>, year_cubed: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn parse_vec(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    a.iter().enumerate().map(|(i, x)| x.as_f64().filter(|z| z.is_finite())
        .ok_or_else(|| format!("{key}[{i}] must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = finite_num(&Value::Object(o.clone()), "n")?;
    if n < 0.0 || n.fract() != 0.0 || n > usize::MAX as f64 { return Err("n must be a nonnegative integer".into()); }
    let n = n as usize;
    let counts = parse_vec(&Value::Object(o.clone()), "C")?;
    let year = parse_vec(&Value::Object(o.clone()), "year")?;
    if counts.len() != n || year.len() != n { return Err("C and year must have length n".into()); }
    for (i, &c) in counts.iter().enumerate() {
        if c < 0.0 || c.fract() != 0.0 { return Err(format!("C[{i}] must be a nonnegative integer")); }
    }
    // This is exactly the original transformed-data calculation, not a new summary.
    let year_squared: Vec<f64> = year.iter().map(|&x| x * x).collect();
    let year_cubed: Vec<f64> = year_squared.iter().zip(&year).map(|(&x2, &x)| x2 * x).collect();
    if year_squared.iter().chain(&year_cubed).any(|x| !x.is_finite()) { return Err("transformed year values are non-finite".into()); }
    Ok(Data { counts, year, year_squared, year_cubed })
}
fn expected_layout(_d: &Data) -> String { "alpha\nbeta1\nbeta2\nbeta3".into() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
// Stan lower/upper scalar transform and Jacobian, then fused poisson_log likelihood.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let lows = [-20.0, -10.0, -10.0, -10.0];
    let widths = [40.0, 20.0, 20.0, 20.0];
    let mut theta = [0.0; 4];
    let mut lp = 0.0;
    for j in 0..4 {
        let s = sigmoid(q[j]);
        theta[j] = lows[j] + widths[j] * s;
        // log |d theta/d q| = log(width) + log(s) + log(1-s), stable for finite q.
        lp += widths[j].ln() - softplus(-q[j]) - softplus(q[j]);
        g[j] = 1.0 - 2.0 * s;
    }
    let (a,b1,b2,b3) = (theta[0],theta[1],theta[2],theta[3]);
    for i in 0..d.counts.len() {
        let eta = a + b1*d.year[i] + b2*d.year_squared[i] + b3*d.year_cubed[i];
        let e = eta.exp();
        if !e.is_finite() { return Err("non-finite Poisson rate".into()); }
        lp += d.counts[i]*eta - e; // propto=true drops log(C!)
        let r = d.counts[i] - e;
        g[0] += r * widths[0] * sigmoid(q[0]) * (1.0-sigmoid(q[0]));
        g[1] += r * d.year[i] * widths[1] * sigmoid(q[1]) * (1.0-sigmoid(q[1]));
        g[2] += r * d.year_squared[i] * widths[2] * sigmoid(q[2]) * (1.0-sigmoid(q[2]));
        g[3] += r * d.year_cubed[i] * widths[3] * sigmoid(q[3]) * (1.0-sigmoid(q[3]));
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?; let data=parse_data(v)?; let want=expected_layout(&data); let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?; if ndim!=4||got!=want{return Err("dimension/layout mismatch".into())}; Ok(Bound{data,ndim})})); match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
