use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies retain raw `dist`; the Stan transformed-data division is done
// in every evaluation, matching the original transformed-data program.
struct Data { n: usize, switched: Vec<f64>, dist: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    v.as_u64().and_then(|x| usize::try_from(x).ok()).ok_or_else(|| format!("{key} must be a nonnegative integer"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let n = integer(o.get("N").ok_or("missing N")?, "N")?;
    let switched_v = o.get("switched").and_then(Value::as_array).ok_or("switched must be an array")?;
    let dist_v = o.get("dist").and_then(Value::as_array).ok_or("dist must be an array")?;
    if switched_v.len() != n || dist_v.len() != n { return Err("data length does not match N".into()); }
    let mut switched = Vec::with_capacity(n); let mut dist = Vec::with_capacity(n);
    for (i, x) in switched_v.iter().enumerate() {
        let y = x.as_i64().ok_or_else(|| format!("switched[{i}] must be integer"))?;
        if y != 0 && y != 1 { return Err(format!("switched[{i}] must be 0 or 1")); }
        switched.push(y as f64);
    }
    for (i, x) in dist_v.iter().enumerate() {
        let z = x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("dist[{i}] must be finite numeric"))?;
        dist.push(z);
    }
    Ok(Data { n, switched, dist })
}
fn expected_layout(_: &Data) -> String { "alpha\nbeta.1".into() }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let (alpha, beta) = (q[0], q[1]);
    let mut lp = 0.0; let mut ga = 0.0; let mut gb = 0.0;
    for i in 0..d.n {
        // Original transformed data: dist100 = dist / 100.0; x = [dist100']'.
        let x = d.dist[i] / 100.0;
        let eta = alpha + x * beta;
        lp += d.switched[i] * eta - softplus(eta);
        let residual = d.switched[i] - sigmoid(eta);
        ga += residual; gb += x * residual;
    }
    g[0] = ga; g[1] = gb; Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=2||got!=expected_layout(&data){return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
