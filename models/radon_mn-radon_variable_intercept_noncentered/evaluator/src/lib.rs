use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn number(v: &Value, name: &str) -> Result<f64,String> { v.as_f64().filter(|x| x.is_finite()).ok_or_else(||format!("{name} must contain finite numbers")) }
fn integer(v: &Value,name:&str)->Result<usize,String> { let x=number(v,name)?; if x<0.0 || x.fract()!=0.0 || x > usize::MAX as f64 {Err(format!("{name} must be nonnegative integer"))} else {Ok(x as usize)} }
fn array<'a>(o:&'a serde_json::Map<String,Value>,name:&str,n:usize)->Result<&'a Vec<Value>,String>{let a=o.get(name).and_then(Value::as_array).ok_or_else(||format!("{name} must be an array"))?;if a.len()!=n {Err(format!("{name} length mismatch"))}else{Ok(a)}}
fn parse_data(v: Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be a JSON object")?;
 let j=integer(o.get("J").ok_or("missing J")?,"J")?; let n=integer(o.get("N").ok_or("missing N")?,"N")?;
 if j==0 {return Err("J must be positive".into());}
 let ci=array(o,"county_idx",n)?; let fm=array(o,"floor_measure",n)?;let yr=array(o,"log_radon",n)?;
 let mut county=Vec::with_capacity(n);let mut floor=Vec::with_capacity(n);let mut y=Vec::with_capacity(n);
 for i in 0..n {let c=integer(&ci[i],"county_idx")?;if c<1||c>j{return Err("county_idx out of bounds".into())};county.push(c-1);floor.push(number(&fm[i],"floor_measure")?);y.push(number(&yr[i],"log_radon")?);}
 Ok(Data{j,n,county,floor,y})
}
fn expected_layout(d:&Data)->String { let mut n=Vec::with_capacity(d.j+4);for i in 1..=d.j{n.push(format!("alpha_raw.{i}"));}n.extend(["beta".into(),"mu_alpha".into(),"sigma_alpha".into(),"sigma_y".into()]);n.join("\n") }
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 if q.iter().any(|x|!x.is_finite()){return Err("non-finite position".into())};g.fill(0.0);
 let beta=q[d.j];let mu=q[d.j+1];let sa=q[d.j+2].exp();let sy=q[d.j+3].exp();
 if !sa.is_finite()||!sy.is_finite(){return Err("non-finite transformed scale".into())}
 let mut lp=-0.5*(d.n as f64)*(2.0*std::f64::consts::PI).ln(); // explicit normal_lpdf retains its normalizing constant
 // Stan normal priors under propto=true, plus lower-bound transform Jacobians.
 for i in 0..d.j {lp-=0.5*q[i]*q[i];g[i]=-q[i];}
 lp-=0.5*(beta/10.0)*(beta/10.0);g[d.j]=-beta/100.0;
 lp-=0.5*(mu/10.0)*(mu/10.0);g[d.j+1]=-mu/100.0;
 lp += -0.5 * sa * sa + q[d.j + 2]; let mut dsa = -sa;
 lp += -0.5 * sy * sy + q[d.j + 3]; let mut dsy = -sy;
 let inv_sy2=1.0/(sy*sy);
 // Fused original per-observation normal_lpdf likelihood; no data-only reductions.
 for i in 0..d.n {let a=d.county[i];let eta=mu+sa*q[a]+d.floor[i]*beta;let r=d.y[i]-eta;lp+=-0.5*r*r*inv_sy2-sy.ln();let de=r*inv_sy2;g[a]+=de*sa;g[d.j]+=de*d.floor[i];g[d.j+1]+=de;dsa+=de*q[a];dsy+=r*r/(sy*sy*sy)-1.0/sy;}
 g[d.j+2]=dsa*sa+1.0;g[d.j+3]=dsy*sy+1.0;Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle]pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let want=expected_layout(&d);if ndim!=d.j+4||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
