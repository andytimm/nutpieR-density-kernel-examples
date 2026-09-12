use serde_json::Value;
use std::{ffi::{c_char,c_void},panic::{catch_unwind,AssertUnwindSafe},slice};

struct Data { n: usize, j: usize, county: Vec<usize>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn finite_num(v:&Value,key:&str)->Result<f64,String>{v.get(key).and_then(Value::as_f64).filter(|x|x.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))}
fn pos_int(v:&Value,key:&str)->Result<usize,String>{let x=finite_num(v,key)?; if x<0.0||x.fract()!=0.0||x>(usize::MAX as f64){Err(format!("{key} must be nonnegative integer"))}else{Ok(x as usize)}}
fn parse_data(v:Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be a JSON object")?; let vv=Value::Object(o.clone());
 let n=pos_int(&vv,"N")?; let j=pos_int(&vv,"J")?; if j==0 {return Err("J must be positive".into())}
 let ca=o.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
 let ya=o.get("log_radon").and_then(Value::as_array).ok_or("log_radon must be an array")?;
 if ca.len()!=n||ya.len()!=n{return Err("data length mismatch".into())}
 let mut county=Vec::with_capacity(n); let mut y=Vec::with_capacity(n);
 for x in ca {let z=x.as_f64().filter(|x|x.is_finite()&&*x>=1.&&x.fract()==0.&&*x<=j as f64).ok_or("county_idx out of bounds")?; county.push(z as usize-1)}
 for x in ya {y.push(x.as_f64().filter(|x|x.is_finite()).ok_or("log_radon must be finite")?)}
 Ok(Data{n,j,county,y})
}
fn expected_layout(d:&Data)->String {(1..=d.j).map(|i|format!("alpha.{i}")).chain(["mu_alpha".into(),"sigma_alpha".into(),"sigma_y".into()]).collect::<Vec<_>>().join("\n")}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 let mu=q[d.j]; let lsa=q[d.j+1]; let lsy=q[d.j+2]; let sa=lsa.exp(); let sy=lsy.exp();
 if !mu.is_finite()||!sa.is_finite()||!sy.is_finite(){return Err("non-finite position".into())}
 g.fill(0.0); let mut lp= -0.5*sa*sa + lsa -0.5*sy*sy + lsy -0.5*(mu/10.0)*(mu/10.0);
 g[d.j+1]=1.0-sa*sa; g[d.j+2]=1.0-sy*sy; g[d.j]=-mu/100.0;
 // Original alpha ~ normal(mu_alpha, sigma_alpha): retain parameter-dependent log scale.
 for i in 0..d.j {let r=(q[i]-mu)/sa; lp += -0.5*r*r-lsa; g[i]+=-r/sa; g[d.j]+=r/sa; g[d.j+1]+=r*r-1.0;}
 // Original transformed/model-block mu[n] = alpha[county_idx[n]] and explicit normal_lpdf.
 // Explicit lpdf retains its full normalizing constant under BridgeStan propto=true.
 const LOG_SQRT_2PI:f64=0.91893853320467274178;
 for n in 0..d.n {let i=d.county[n]; let r=(d.y[n]-q[i])/sy; lp += -0.5*r*r-lsy-LOG_SQRT_2PI; g[i]+=r/sy; g[d.j+2]+=r*r-1.0;}
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let want=expected_layout(&data);if ndim!=data.j+3||got!=want{return Err("dimension/layout mismatch".into())}Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
