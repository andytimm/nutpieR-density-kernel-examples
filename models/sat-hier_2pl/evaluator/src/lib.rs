use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe}, slice};

struct Data { i: usize, j: usize, n: usize, ii: Vec<usize>, jj: Vec<usize>, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn int(v:&Value,k:&str)->Result<usize,String>{ v.get(k).and_then(Value::as_u64).filter(|&x|x>0).map(|x|x as usize).ok_or_else(||format!("{k} must be a positive integer")) }
fn ints(v:&Value,k:&str,n:usize,max:usize)->Result<Vec<usize>,String>{let a=v.get(k).and_then(Value::as_array).ok_or_else(||format!("{k} must be an array"))?;if a.len()!=n{return Err(format!("{k} length mismatch"))}; a.iter().map(|x|x.as_u64().filter(|&z|z>0 && z<=max as u64).map(|z|z as usize-1).ok_or_else(||format!("{k} out of bounds"))).collect()}
fn bits(v:&Value,n:usize)->Result<Vec<u8>,String>{let a=v.get("y").and_then(Value::as_array).ok_or("y must be an array")?;if a.len()!=n{return Err("y length mismatch".into())};a.iter().map(|x|x.as_u64().filter(|&z|z<=1).map(|z|z as u8).ok_or_else(||"y must contain 0/1".to_string())).collect()}
fn parse_data(v:Value)->Result<Data,String>{let o=v.as_object().ok_or("data must be a JSON object")?;let v=Value::Object(o.clone());let i=int(&v,"I")?;let j=int(&v,"J")?;let n=int(&v,"N")?;Ok(Data{i,j,n,ii:ints(&v,"ii",n,i)?,jj:ints(&v,"jj",n,j)?,y:bits(&v,n)?})}
fn expected_layout(d:&Data)->String{let mut a=Vec::with_capacity(d.j+2*d.i+5);for k in 1..=d.j{a.push(format!("theta.{k}"))}for name in ["xi1","xi2"]{for k in 1..=d.i{a.push(format!("{name}.{k}"))}}for k in 1..=2{a.push(format!("mu.{k}"))}for k in 1..=2{a.push(format!("tau.{k}"))}a.push("L_Omega.1".into());a.join("\n")}
fn sigmoid(x:f64)->f64{if x>=0.0{1.0/(1.0+(-x).exp())}else{let z=x.exp();z/(1.0+z)}}
fn softplus(x:f64)->f64{x.max(0.0)+(-x.abs()).exp().ln_1p()}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 g.fill(0.0); let j=d.j;let i=d.i;let x1=j;let x2=j+i;let mu=x2+i;let tq=mu+2;let zix=tq+2;
 let t1=q[tq].exp(); let t2=q[tq+1].exp();
 // Pinned Stan Math corr_constrain(z) = tanh(z).
 let r=q[zix].tanh(); let one=1.0-r*r; let scale=one.sqrt();
 if !(one>0.0 && t1.is_finite() && t2.is_finite()){return Err("transform overflow".into())};
 let log_s=0.5*one.ln(); let mut lp=0.0;
 let mut gt1=0.0;let mut gt2=0.0;let mut gr=0.0;
 // L_Sigma = diag_pre_multiply(tau, [[1,0],[r,sqrt(1-r^2)]]) exactly.
 for k in 0..i {
   let a=(q[x1+k]-q[mu])/t1;
   let b=(q[x2+k]-q[mu+1])/t2;
   let v=(b-r*a)/scale;
   lp+=-0.5*(a*a+v*v)-t1.ln()-t2.ln()-log_s;
   let da=-a+r*v/scale;
   let db=-v/scale;
   g[x1+k]+=da/t1; g[x2+k]+=db/t2;
   g[mu]-=da/t1; g[mu+1]-=db/t2;
   gt1+=da*(-a/t1)-1.0/t1; gt2+=db*(-b/t2)-1.0/t2;
   gr+=v*a/scale - r*v*v/one + r/one;
 }
 // Explicit multi_normal_cholesky_lpdf retains its normalizer under propto=true.
 // Equivalent -I * log(2*pi), written plainly to prevent any target offset calibration.
 lp-=i as f64 * (2.0*std::f64::consts::PI).ln();
 for n in 0..d.n {let k=d.ii[n];let person=d.jj[n];let alpha=q[x1+k].exp();let delta=q[person]-q[x2+k];let eta=alpha*delta;let c=d.y[n] as f64-sigmoid(eta);lp+=(d.y[n] as f64)*eta-softplus(eta);g[person]+=alpha*c;g[x2+k]-=alpha*c;g[x1+k]+=eta*c;}
 for k in 0..j {lp+=-0.5*q[k]*q[k];g[k]+=-q[k];}lp+=-0.5*q[mu]*q[mu]-0.5*(q[mu+1]/5.0).powi(2);g[mu]+=-q[mu];g[mu+1]+=-q[mu+1]/25.0;
 lp+=-0.1*t1-0.1*t2;gt1+=-0.1;gt2+=-0.1;
 // LKJ(4) factor and corr_cholesky_constrain Jacobian, in Stan transform order.
 lp+=6.0*log_s+one.ln(); gr+=-8.0*r/one;
 lp+=q[tq]+q[tq+1];g[tq]=t1*gt1+1.0;g[tq+1]=t2*gt2+1.0;
 // Pinned corr_constrain has dr/dz = 1-r^2.
 g[zix]=one*gr;
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,c:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,c,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;let n=data.j+2*data.i+5;if ndim!=n||got!=want{Err("dimension/layout mismatch".into())}else{Ok(Bound{data,ndim})}}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){Err("null bound handle".into())}else{Ok(Box::into_raw(Box::new(Workspace)).cast())}}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gp:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||gp.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())}let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gp,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
