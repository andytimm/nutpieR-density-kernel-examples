use serde_json::Value;
use std::{ffi::{c_char,c_void},panic::{catch_unwind,AssertUnwindSafe},slice};
const D:usize=6; const K:usize=2;
struct Data { n:usize,u:Vec<f64>,v:Vec<f64>,alpha:[[f64;K];K],tau:f64,rho:f64 }
struct Bound { data:Data, ndim:usize }
struct Workspace { prev:[AD;K], cur:[AD;K] }
#[derive(Clone,Copy)] struct AD { v:f64, d:[f64;D] }
impl AD { fn c(v:f64)->Self{Self{v,d:[0.;D]}} fn add(self,b:Self)->Self{let mut d=[0.;D];for i in 0..D{d[i]=self.d[i]+b.d[i]}Self{v:self.v+b.v,d}} fn log(self)->Self{let mut d=self.d;for x in &mut d{*x/=self.v}Self{v:self.v.ln(),d}} }
fn num(v:&Value,k:&str)->Result<f64,String>{v.get(k).and_then(Value::as_f64).filter(|x|x.is_finite()).ok_or_else(||format!("{k} must be finite numeric"))}
fn usize_field(v:&Value,k:&str)->Result<usize,String>{let x=num(v,k)?;if x>=1.&&x.fract()==0. {Ok(x as usize)}else{Err(format!("{k} must be positive integer"))}}
fn vector(v:&Value,k:&str,n:usize)->Result<Vec<f64>,String>{let a=v.get(k).and_then(Value::as_array).ok_or_else(||format!("{k} must be array"))?;if a.len()!=n{return Err(format!("{k} length"))}a.iter().map(|x|x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{k} finite"))).collect()}
fn lgamma(z:f64)->f64 { // Lanczos, valid for positive alpha and alpha sums
 const C:[f64;9]=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
 if z < 0.5 { return std::f64::consts::PI.ln()-(std::f64::consts::PI*z).sin().ln()-lgamma(1.0-z); }
 let x=z-1.0; let mut a=C[0]; for i in 1..C.len(){a+=C[i]/(x+i as f64);} let q=x+7.5;
 0.5*(2.0*std::f64::consts::PI).ln()+(x+0.5)*q.ln()-q+a.ln()
}
fn parse_data(v:Value)->Result<Data,String>{let o=v.as_object().ok_or("data must object")?;let vv=Value::Object(o.clone());let n=usize_field(&vv,"N")?;if usize_field(&vv,"K")?!=2{return Err("this Stan program requires K=2".into())};let u=vector(&vv,"u",n)?;let x=vector(&vv,"v",n)?;let tau=num(&vv,"tau")?;let rho=num(&vv,"rho")?;if tau<=0.||rho<=0.{return Err("tau and rho must be positive".into())};let a=vv.get("alpha").and_then(Value::as_array).ok_or("alpha matrix")?;if a.len()!=2{return Err("alpha rows".into())};let mut alpha=[[0.;K];K];for i in 0..K {let r=a[i].as_array().ok_or("alpha row")?;if r.len()!=2{return Err("alpha columns".into())}for j in 0..K{alpha[i][j]=r[j].as_f64().filter(|z|z.is_finite()&&*z>=0.).ok_or("alpha must be nonnegative finite")?}}Ok(Data{n,u,v:x,alpha,tau,rho})}
fn layout()->String{"theta1.1\ntheta2.1\nphi.1\nphi.2\nlambda.1\nlambda.2".into()}
fn sig(x:f64)->f64{if x>=0.{1./(1.+(-x).exp())}else{let e=x.exp();e/(1.+e)}}
fn lse(a:AD,b:AD)->AD{let m=a.v.max(b.v);let ea=(a.v-m).exp();let eb=(b.v-m).exp();let z=ea+eb;let mut d=[0.;D];for i in 0..D{d[i]=(ea*a.d[i]+eb*b.d[i])/z}AD{v:m+z.ln(),d}}
fn normal(y:f64,mu:AD,s:f64)->AD{let r=(y-mu.v)/s;let mut d=mu.d;for x in &mut d{*x*=r/s}AD{v:-0.5*r*r-s.ln()-0.5*(2.0*std::f64::consts::PI).ln(),d}}
fn eval_model(x:&Data,q:&[f64],g:&mut[f64],w:&mut Workspace)->Result<f64,String>{
 if q.iter().any(|z|!z.is_finite()){return Err("nonfinite position".into())};g.fill(0.);
 let simplex_scale=2.0_f64.sqrt(); let p=[sig(simplex_scale*q[0]),sig(simplex_scale*q[1])]; let theta=[[p[0],1.-p[0]],[p[1],1.-p[1]]];
 let mut phi=[AD::c(q[2]),AD::c(q[2]+q[3].exp())];phi[0].d[2]=1.;phi[1].d[2]=1.;phi[1].d[3]=q[3].exp();
 let mut lam=[AD::c(q[4]),AD::c(q[4]+q[5].exp())];lam[0].d[4]=1.;lam[1].d[4]=1.;lam[1].d[5]=q[5].exp();
 let mut lp=0.; let mut gd=[0.;D];
 // Explicit Stan lpdf calls retain their parameter-independent normalizers under propto=true.
 for r in 0..K{let sum=x.alpha[r][0]+x.alpha[r][1];lp+=lgamma(sum)-lgamma(x.alpha[r][0])-lgamma(x.alpha[r][1]);for j in 0..K{lp+=(x.alpha[r][j]-1.)*theta[r][j].ln();let dp=if j==0 {simplex_scale*p[r]*(1.-p[r])} else {-simplex_scale*p[r]*(1.-p[r])};gd[r]+=(x.alpha[r][j]-1.)*dp/theta[r][j];}}
 for k in 0..K {let m=if k==0 {0.}else{3.};let z=phi[k].v-m;lp+=-0.5*z*z-0.5*(2.*std::f64::consts::PI).ln();for i in 0..D{gd[i]+=-z*phi[k].d[i]};let z=lam[k].v-m;lp+=-0.5*z*z-0.5*(2.*std::f64::consts::PI).ln();for i in 0..D{gd[i]+=-z*lam[k].d[i]}}
 // Jacobians: two K=2 simplex transforms and two ordered transforms.
 for r in 0..K {lp+=0.5*2.0_f64.ln()+p[r].ln()+(1.-p[r]).ln();gd[r]+=simplex_scale*(1.-2.*p[r])} lp+=q[3]+q[5];gd[3]+=1.;gd[5]+=1.;
 for k in 0..K {w.prev[k]=normal(x.u[0],phi[k],x.tau).add(normal(x.v[0],lam[k],x.rho));}
 for t in 1..x.n {for k in 0..K {let e=normal(x.u[t],phi[k],x.tau).add(normal(x.v[t],lam[k],x.rho));let mut a=w.prev[0];a.v+=theta[0][k].ln();a.d[0]+=if k==0 {simplex_scale*(1.-p[0])} else {-simplex_scale*p[0]};let mut b=w.prev[1];b.v+=theta[1][k].ln();b.d[1]+=if k==0 {simplex_scale*(1.-p[1])} else {-simplex_scale*p[1]};w.cur[k]=lse(a,b).add(e);}std::mem::swap(&mut w.prev,&mut w.cur);}
 let z=lse(w.prev[0],w.prev[1]);lp+=z.v;for i in 0..D{g[i]=gd[i]+z.d[i]}Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0}}}
fn fatal(p:*mut c_char,c:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,c,s.as_ref())};2}
#[no_mangle]pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,lay:*const c_char,lay_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let got=std::str::from_utf8(unsafe{bytes(lay,lay_len)?}).map_err(|_|"layout utf8")?;if ndim!=D||got!=layout(){return Err("dimension/layout mismatch".into())}Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())}let z=AD::c(0.);Ok(Box::into_raw(Box::new(Workspace{prev:[z;K],cur:[z;K]})).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,n:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if n!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,n)};let g=unsafe{slice::from_raw_parts_mut(g,n)};let v=eval_model(&b.data,q,g,unsafe{&mut*w.cast::<Workspace>()})?;if !v.is_finite()||g.iter().any(|z|!z.is_finite()){return Ok(1)}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
