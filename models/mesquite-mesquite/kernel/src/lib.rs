use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe}, slice};

struct Data { n: usize, weight: Vec<f64>, x: [Vec<f64>; 6] }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn numeric_array(o: &serde_json::Map<String,Value>, key: &str, n: usize) -> Result<Vec<f64>,String> {
 let a=o.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be numeric array"))?;
 if a.len()!=n {return Err(format!("{key} length mismatch"));}
 a.iter().map(|v| v.as_f64().filter(|x|x.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be a JSON object")?;
 let n=o.get("N").and_then(Value::as_u64).ok_or("N must be nonnegative integer")? as usize;
 if n == 0 {return Err("N must be positive".into());}
 Ok(Data {n, weight:numeric_array(o,"weight",n)?, x:[numeric_array(o,"diam1",n)?,numeric_array(o,"diam2",n)?,numeric_array(o,"canopy_height",n)?,numeric_array(o,"total_height",n)?,numeric_array(o,"density",n)?,numeric_array(o,"group",n)?]})
}
fn expected_layout()->String { ["beta.1","beta.2","beta.3","beta.4","beta.5","beta.6","beta.7","sigma"].join("\n") }
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 if q.iter().any(|x|!x.is_finite()) {return Err("non-finite position".into());}
 let sigma=q[7].exp();
 if !sigma.is_finite() || sigma==0.0 {return Err("sigma transform overflow/underflow".into());}
 let inv=1.0/sigma; let inv2=inv*inv;
 g.fill(0.0); let mut lp=q[7]; let mut gsigma=1.0-(d.n as f64);
 for i in 0..d.n { let mut mu=q[0]; for j in 0..6 {mu+=q[j+1]*d.x[j][i];} let r=d.weight[i]-mu; lp += -q[7]-0.5*r*r*inv2; g[0]+=r*inv2; for j in 0..6 {g[j+1]+=r*d.x[j][i]*inv2;} gsigma+=r*r*inv2; }
 g[7]=gsigma; Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=8||got!=expected_layout(){return Err("dimension/layout mismatch".into())}Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())}let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};match eval_model(&b.data,q,g){Ok(v) if v.is_finite()&&g.iter().all(|x|x.is_finite())=>{unsafe{*lp=v};Ok(0)},Ok(_)=>Ok(1),Err(_)=>Ok(1)}}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
