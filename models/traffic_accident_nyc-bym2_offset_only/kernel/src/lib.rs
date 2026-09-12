use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe}, slice};

struct Data { n: usize, node1: Vec<usize>, node2: Vec<usize>, y: Vec<f64>, log_e: Vec<f64>, scaling: f64 }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn number(o: &serde_json::Map<String,Value>, key:&str)->Result<f64,String>{o.get(key).and_then(Value::as_f64).filter(|x|x.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))}
fn count(o:&serde_json::Map<String,Value>,key:&str)->Result<usize,String>{let x=number(o,key)?; if x<0.0||x.fract()!=0.0||x>(usize::MAX as f64){Err(format!("{key} must be a nonnegative integer"))}else{Ok(x as usize)}}
fn array(o:&serde_json::Map<String,Value>,key:&str,n:usize)->Result<Vec<f64>,String>{let a=o.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be numeric array"))?;if a.len()!=n{return Err(format!("{key} length mismatch"))};a.iter().map(|x|x.as_f64().filter(|v|v.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))).collect()}
fn index_array(o:&serde_json::Map<String,Value>,key:&str,n:usize,upper:usize)->Result<Vec<usize>,String>{let a=array(o,key,n)?; a.into_iter().map(|x|if x<1.||x.fract()!=0.||x>(upper as f64){Err(format!("{key} index out of bounds"))}else{Ok(x as usize-1)}).collect()}
fn parse_data(v:Value)->Result<Data,String>{let o=v.as_object().ok_or("data must be a JSON object")?;let n=count(o,"N")?;let ne=count(o,"N_edges")?; if n==0{return Err("N must be positive".into())};let e=array(o,"E",n)?;if e.iter().any(|x|*x<0.){return Err("E must be nonnegative".into())}; let log_e:Vec<f64>=e.into_iter().map(|x|x.ln()).collect(); if log_e.iter().any(|x|!x.is_finite()){return Err("E must be positive for transformed data log(E)".into())};let y=array(o,"y",n)?;if y.iter().any(|x|*x<0.||x.fract()!=0.){return Err("y must be nonnegative integers".into())};let scaling=number(o,"scaling_factor")?;if scaling<=0.{return Err("scaling_factor must be positive".into())};let node1=index_array(o,"node1",ne,n)?;let node2=index_array(o,"node2",ne,n)?;Ok(Data{n,node1,node2,y,log_e,scaling})}
fn expected_layout(d:&Data)->String{let mut x=vec!["beta0".to_string(),"sigma".to_string(),"rho".to_string()];for p in ["theta","phi"]{for i in 1..=d.n{x.push(format!("{p}.{i}"));}}x.join("\n")}
fn sigmoid(x:f64)->f64{if x>=0.{1./(1.+(-x).exp())}else{let z=x.exp();z/(1.+z)}}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 let n=d.n; let beta=q[0];let sigma=q[1].exp();let rho=sigmoid(q[2]);let a=(1.-rho).sqrt();let c=(rho/d.scaling).sqrt();let theta=&q[3..3+n];let phi=&q[3+n..3+2*n];
 g.fill(0.); let mut lp=q[1]+rho.ln()+(1.-rho).ln(); // transform Jacobians
 lp += -0.5*beta*beta -0.5*sigma*sigma -0.5*rho.ln()-0.5*(1.-rho).ln();
 let mut db=-beta;let mut ds=-sigma;let mut dr=-0.5/rho+0.5/(1.-rho);
 let mut sumphi=0.; for &x in phi {sumphi+=x}; let zsum=sumphi/(0.001*n as f64); lp+=-0.5*zsum*zsum; let sumgrad=-sumphi/(0.001*n as f64).powi(2);
 for i in 0..n {let conv=a*theta[i]+c*phi[i]; let eta=d.log_e[i]+beta+sigma*conv;let ex=eta.exp(); if !ex.is_finite(){return Err("nonfinite poisson intensity".into())};let r=d.y[i]-ex;lp+=d.y[i]*eta-ex; db+=r;ds+=r*conv;dr+=r*sigma*(-0.5*theta[i]/a+0.5*c*phi[i]/rho);g[3+i]+=r*sigma*a-theta[i];g[3+n+i]+=r*sigma*c+sumgrad;lp+=-0.5*theta[i]*theta[i];}
 for e in 0..d.node1.len(){let i=d.node1[e];let j=d.node2[e];let dif=phi[i]-phi[j];lp+=-0.5*dif*dif;g[3+n+i]-=dif;g[3+n+j]+=dif;}
 g[0]=db;g[1]=ds*sigma+1.;g[2]=(dr+1./rho-1./(1.-rho))*rho*(1.-rho);
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle]pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let ans=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=3+2*data.n||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match ans{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){Err("null bound handle".into())}else{Ok(Box::into_raw(Box::new(Workspace)).cast())}}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(v))=>v,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
