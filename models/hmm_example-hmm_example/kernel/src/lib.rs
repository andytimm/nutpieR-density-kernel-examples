use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe},slice};

struct Data { n: usize, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace { gamma: Vec<f64>, adj: Vec<f64> }

fn parse_data(v: Value) -> Result<Data,String> {
 let o=v.as_object().ok_or("data must be a JSON object")?;
 let n=o.get("N").and_then(Value::as_u64).ok_or("N must be a nonnegative integer")? as usize;
 let k=o.get("K").and_then(Value::as_u64).ok_or("K must be integer")?;
 if k != 2 { return Err("this Stan model requires K = 2".into()); }
 if n == 0 { return Err("N must be positive for y[1]".into()); }
 let a=o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
 if a.len()!=n { return Err("y length must equal N".into()); }
 let mut y=Vec::with_capacity(n);
 for x in a { y.push(x.as_f64().filter(|z|z.is_finite()).ok_or("y must be finite numeric")?); }
 Ok(Data{n,y})
}
fn expected_layout(_: &Data)->String { "theta1.1\ntheta2.1\nmu.1\nmu.2".into() }
fn sigmoid(x:f64)->f64 { if x>=0.0 { 1.0/(1.0+(-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn log_sigmoid(x:f64)->f64 { if x>=0.0 {-(-x).exp().ln_1p()} else {x- x.exp().ln_1p()} }
fn logsum(a:f64,b:f64)->f64 { if a>b {a+(b-a).exp().ln_1p()} else {b+(a-b).exp().ln_1p()} }
fn eval_model(d:&Data, w:&mut Workspace, q:&[f64], g:&mut[f64])->Result<f64,String> {
 if q.iter().any(|x|!x.is_finite()) {return Err("non-finite position".into());}
 let simplex_scale=2.0_f64.sqrt(); let u1=simplex_scale*q[0]; let u2=simplex_scale*q[1]; let p1=sigmoid(u1); let p2=sigmoid(u2);
 let theta=[[p1,1.0-p1],[p2,1.0-p2]];
 let ltheta=[[log_sigmoid(u1),log_sigmoid(-u1)],[log_sigmoid(u2),log_sigmoid(-u2)]];
 let e0=q[2].exp(); let e1=q[3].exp(); if !e0.is_finite()||!e1.is_finite(){return Err("positive_ordered transform overflow".into());}
 let mu=[e0,e0+e1]; if !mu[1].is_finite(){return Err("positive_ordered transform overflow".into());}
 let emit=|t:usize,k:usize| -> f64 { let r=d.y[t]-mu[k]; -0.5*r*r-0.5*std::f64::consts::TAU.ln() };
 for k in 0..2 {w.gamma[k]=emit(0,k);}
 for t in 1..d.n {for k in 0..2 {let a=w.gamma[(t-1)*2]+ltheta[0][k]; let b=w.gamma[(t-1)*2+1]+ltheta[1][k]; w.gamma[t*2+k]=emit(t,k)+logsum(a,b);}}
 let mut lp=logsum(w.gamma[(d.n-1)*2],w.gamma[(d.n-1)*2+1]);
 // Explicit normal_lpdf calls retain their normalizers under propto=true.
 for k in 0..2 {let r=mu[k]-if k==0{3.0}else{10.0}; lp += -0.5*r*r-0.5*std::f64::consts::TAU.ln();}
 // simplex and positive_ordered Jacobians
 lp += ltheta[0][0]+ltheta[0][1]+ltheta[1][0]+ltheta[1][1]+std::f64::consts::LN_2+q[2]+q[3];
 w.adj.fill(0.0); let end=(d.n-1)*2; let den=logsum(w.gamma[end],w.gamma[end+1]); w.adj[end]=(w.gamma[end]-den).exp(); w.adj[end+1]=(w.gamma[end+1]-den).exp();
 let mut dmu=[-(mu[0]-3.0),-(mu[1]-10.0)]; let mut dlogtheta=[[0.0;2];2];
 for t in (1..d.n).rev() { for k in 0..2 { let aidx=t*2+k; let ad=w.adj[aidx]; let a=w.gamma[(t-1)*2]+ltheta[0][k]; let b=w.gamma[(t-1)*2+1]+ltheta[1][k]; let z=logsum(a,b); let wa=(a-z).exp()*ad; let wb=(b-z).exp()*ad; w.adj[(t-1)*2]+=wa; w.adj[(t-1)*2+1]+=wb; dlogtheta[0][k]+=wa; dlogtheta[1][k]+=wb; dmu[k]+=ad*(d.y[t]-mu[k]); }}
 for k in 0..2 {dmu[k]+=w.adj[k]*(d.y[0]-mu[k]);}
 // stick-breaking simplex derivative and log-Jacobian derivative.
 g[0]=simplex_scale*(dlogtheta[0][0]*(1.0-p1)-dlogtheta[0][1]*p1 + 1.0-2.0*p1);
 g[1]=simplex_scale*(dlogtheta[1][0]*(1.0-p2)-dlogtheta[1][1]*p2 + 1.0-2.0*p2);
 g[2]=(dmu[0]+dmu[1])*e0+1.0; g[3]=dmu[1]*e1+1.0;
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=4||got!=expected_layout(&d){return Err("dimension/layout mismatch".into())}Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};let n=unsafe{(&*b.cast::<Bound>()).data.n};Ok(Box::into_raw(Box::new(Workspace{gamma:vec![0.;2*n],adj:vec![0.;2*n]})).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let bb=unsafe{&*b.cast::<Bound>()};if ndim!=bb.ndim||q.is_null(){return Err("evaluation dimension".into())};let qq=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let ww=unsafe{&mut*w.cast::<Workspace>()};let v=eval_model(&bb.data,ww,qq,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
