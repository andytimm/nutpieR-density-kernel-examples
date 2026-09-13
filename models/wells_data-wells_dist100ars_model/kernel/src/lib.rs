use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, switched: Vec<f64>, dist: Vec<f64>, arsenic: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn array_num(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|v| v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let nf = finite_num(&Value::Object(o.clone()), "N")?;
    if nf < 0.0 || nf.fract() != 0.0 || nf > usize::MAX as f64 { return Err("N must be a nonnegative integer".into()); }
    let n = nf as usize;
    let switched = array_num(o, "switched", n)?;
    if switched.iter().any(|&y| y != 0.0 && y != 1.0) { return Err("switched must contain 0 or 1".into()); }
    let dist = array_num(o, "dist", n)?;
    let arsenic = array_num(o, "arsenic", n)?;
    Ok(Data { n, switched, dist, arsenic })
}
fn expected_layout(_d: &Data) -> String { "alpha\nbeta.1\nbeta.2".into() }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let (alpha, b_dist, b_arsenic) = (q[0], q[1], q[2]);
    if !alpha.is_finite() || !b_dist.is_finite() || !b_arsenic.is_finite() { return Err("non-finite position".into()); }
    let mut lp=0.0; let mut ga=0.0; let mut gb0=0.0; let mut gb1=0.0;
    for i in 0..d.n {
        // This exactly reproduces transformed data x=[(dist / 100)', arsenic']'.
        let x0=d.dist[i]/100.0; let x1=d.arsenic[i];
        let eta=alpha+b_dist*x0+b_arsenic*x1;
        let y=d.switched[i];
        lp += y*eta-softplus(eta); // bernoulli_logit_lpmf, propto=true
        let r=y-sigmoid(eta); ga+=r; gb0+=r*x0; gb1+=r*x1;
    }
    g[0]=ga; g[1]=gb0; g[2]=gb1;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())} else {Ok(unsafe{slice::from_raw_parts(p.cast(),n)})} }
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{
 if out.is_null(){return fatal(err,cap,"null bound output");} unsafe{*out=std::ptr::null_mut();}
 let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=3||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));
 match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
