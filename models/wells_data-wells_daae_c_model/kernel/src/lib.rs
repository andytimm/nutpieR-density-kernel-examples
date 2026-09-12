use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// These are the model's original transformed-data columns, copied at bind time.
// `x` is constructed exactly as in the Stan transformed-data block.
struct Data { n: usize, y: Vec<u8>, x: Vec<[f64; 5]> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
    // Reject values which cannot round-trip to usize.
    if n as u64 != o.get("N").and_then(Value::as_u64).unwrap() { return Err("N is too large".into()); }
    let switched = o.get("switched").and_then(Value::as_array).ok_or("switched must be an integer array")?;
    if switched.len() != n { return Err("switched length mismatch".into()); }
    let mut y = Vec::with_capacity(n);
    for z in switched { match z.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("switched values must be 0 or 1".into()) } }
    let dist = number_array(o, "dist", n)?;
    let arsenic = number_array(o, "arsenic", n)?;
    let assoc = number_array(o, "assoc", n)?;
    let educ = number_array(o, "educ", n)?;
    // Original transformed data: vector means and five-column matrix x.
    let md = if n == 0 { 0.0 } else { dist.iter().sum::<f64>() / n as f64 };
    let ma = if n == 0 { 0.0 } else { arsenic.iter().sum::<f64>() / n as f64 };
    if !md.is_finite() || !ma.is_finite() { return Err("transformed-data mean is non-finite".into()); }
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let cd = (dist[i] - md) / 100.0;
        let ca = arsenic[i] - ma;
        let row = [cd, ca, cd * ca, assoc[i], educ[i] / 4.0];
        if row.iter().any(|z| !z.is_finite()) { return Err("transformed data is non-finite".into()); }
        x.push(row);
    }
    Ok(Data { n, y, x })
}
fn expected_layout() -> &'static str { "alpha\nbeta.1\nbeta.2\nbeta.3\nbeta.4\nbeta.5" }
fn softplus(z: f64) -> f64 { z.max(0.0) + (-z.abs()).exp().ln_1p() }
fn sigmoid(z: f64) -> f64 { if z >= 0.0 { 1.0 / (1.0 + (-z).exp()) } else { let e=z.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    g.fill(0.0);
    let mut lp = 0.0;
    for i in 0..d.n {
        let r = &d.x[i];
        let eta = q[0] + q[1]*r[0] + q[2]*r[1] + q[3]*r[2] + q[4]*r[3] + q[5]*r[4];
        let yi = d.y[i] as f64;
        lp += yi * eta - softplus(eta);
        let residual = yi - sigmoid(eta);
        g[0] += residual;
        for j in 0..5 { g[j+1] += residual * r[j]; }
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?; let data=parse_data(v)?; let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?; if ndim!=6 || got!=expected_layout(){return Err("dimension/layout mismatch".into())}; Ok(Bound{data,ndim})})); match answer {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let qq=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};if qq.iter().any(|z|!z.is_finite()){return Ok(1)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,qq,g)?;if !value.is_finite()||g.iter().any(|z|!z.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
