use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe}, slice};

struct Data { t:Vec<f64>, cap:Vec<f64>, y:Vec<f64>, tc:Vec<f64>, x:Vec<f64>, sigmas:Vec<f64>, tau:f64, trend:i64, sa:Vec<f64>, sm:Vec<f64>, a:Vec<u8>, n:usize, kdim:usize, s:usize }
struct Bound { data:Data, ndim:usize }
struct Workspace;
fn num(v:&Value, key:&str)->Result<f64,String>{v.get(key).and_then(Value::as_f64).filter(|x|x.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))}
fn integer(v:&Value,key:&str)->Result<usize,String>{ let x=num(v,key)?; if x<0. || x.fract()!=0. || x>usize::MAX as f64 {Err(format!("{key} must be a nonnegative integer"))} else {Ok(x as usize)} }
fn vec(v:&Value,key:&str,n:usize)->Result<Vec<f64>,String>{let a=v.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be numeric array"))?; if a.len()!=n{return Err(format!("{key} length mismatch"))}; a.iter().map(|x|x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))).collect()}
fn parse_data(v:Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be a JSON object")?; let v=&Value::Object(o.clone()); let n=integer(v,"T")?; let k=integer(v,"K")?; let s=integer(v,"S")?; if n==0||k==0{return Err("T and K must be positive".into())}
 let t=vec(v,"t",n)?; let cap=vec(v,"cap",n)?; let y=vec(v,"y",n)?; let tc=vec(v,"t_change",s)?; let sigmas=vec(v,"sigmas",k)?; if sigmas.iter().any(|x|*x<=0.) {return Err("sigmas must be positive".into())}; let tau=num(v,"tau")?; if tau<=0.{return Err("tau must be positive".into())}; let sa=vec(v,"s_a",k)?;let sm=vec(v,"s_m",k)?;
 let trend=num(v,"trend_indicator")?; if trend.fract()!=0. || !(trend==0.||trend==1.) {return Err("trend_indicator must be 0 or 1".into())};
 let xa=v.get("X").and_then(Value::as_array).ok_or("X must be matrix")?; if xa.len()!=n{return Err("X row mismatch".into())}; let mut x=Vec::with_capacity(n*k); for r in xa {let row=r.as_array().ok_or("X must be matrix")?;if row.len()!=k{return Err("X column mismatch".into() )};for z in row{x.push(z.as_f64().filter(|q|q.is_finite()).ok_or("X must be finite numeric")?)}}
 // Original transformed-data get_changepoint_matrix work, copied to immutable bind state.
 let mut a=vec![0u8;n*s]; let mut cp=0; for i in 0..n {while cp<s && t[i]>=tc[cp]{cp+=1} for j in 0..cp {a[i*s+j]=1}}
 Ok(Data{t,cap,y,tc,x,sigmas,tau,trend:trend as i64,sa,sm,a,n,kdim:k,s})
}
fn expected_layout(d:&Data)->String {let mut z=vec!["k".to_string(),"m".to_string()];for i in 1..=d.s{z.push(format!("delta.{i}"))};z.push("sigma_obs".into());for i in 1..=d.kdim{z.push(format!("beta.{i}"))};z.join("\n")}
fn sigmoid(x:f64)->f64 {if x>=0. {1./(1.+(-x).exp())} else {let e=x.exp();e/(1.+e)}}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 if q.iter().any(|z|!z.is_finite()){return Err("nonfinite position".into())}; g.fill(0.); let s=d.s; let sigma=q[2+s].exp(); if !sigma.is_finite()||sigma==0. {return Err("invalid sigma".into())}; let b0=3+s;
 // positive_constrain(sigma_obs, lp__) adds the unconstrained log-Jacobian.
 let mut lp=q[2+s]-0.5*(q[0]/5.).powi(2)-0.5*(q[1]/5.).powi(2); g[0]=-q[0]/25.;g[1]=-q[1]/25.;
 for j in 0..s {lp-=q[2+j].abs()/d.tau;g[2+j]=if q[2+j]>0.{-1./d.tau}else if q[2+j]<0.{1./d.tau}else{0.};}
 lp-=2.*sigma*sigma; g[2+s]=1.-4.*sigma*sigma;
 for j in 0..d.kdim {let b=q[b0+j];lp-=0.5*(b/d.sigmas[j]).powi(2);g[b0+j]=-b/(d.sigmas[j]*d.sigmas[j]);}
 // gamma and its derivatives, used only by original logistic branch.
 let p=s+2; let mut gamma=vec![0.;s]; let mut dg=vec![vec![0.;p];s]; if d.trend==1 {let mut ks=q[0]; let mut mpr=q[1]; let mut dks=vec![0.;p]; dks[0]=1.;let mut dmpr=vec![0.;p];dmpr[1]=1.;for j in 0..s {let prev=ks;let dprev=dks.clone();ks+=q[2+j];dks[2+j]+=1.;let h=1.-prev/ks; gamma[j]=(d.tc[j]-mpr)*h;for z in 0..p {let dh=-(dprev[z]*ks-prev*dks[z])/(ks*ks);dg[j][z]= -dmpr[z]*h+(d.tc[j]-mpr)*dh;}mpr+=gamma[j];for z in 0..p{dmpr[z]+=dg[j][z];}}}
 for i in 0..d.n {let mut add=0.;let mut mult=0.;for j in 0..d.kdim{let b=q[b0+j];let xx=d.x[i*d.kdim+j];add+=xx*b*d.sa[j];mult+=xx*b*d.sm[j]};let factor=1.+mult;let (trend,dt)=if d.trend==0 {let mut rate=q[0];let mut off=q[1];for j in 0..s {if d.a[i*s+j]!=0 {rate+=q[2+j];off-=d.tc[j]*q[2+j];}}let tr=rate*d.t[i]+off;let mut dd=vec![0.;p];dd[0]=d.t[i];dd[1]=1.;for j in 0..s{if d.a[i*s+j]!=0 {dd[2+j]=d.t[i]-d.tc[j]}};(tr,dd)} else {let mut rate=q[0];let mut off=q[1];let mut dr=vec![0.;p];let mut doff=vec![0.;p];dr[0]=1.;doff[1]=1.;for j in 0..s{if d.a[i*s+j]!=0 {rate+=q[2+j];dr[2+j]+=1.;off+=gamma[j];for z in 0..p{doff[z]+=dg[j][z]}}}let u=rate*(d.t[i]-off);let ss=sigmoid(u);let tr=d.cap[i]*ss;let scale=d.cap[i]*ss*(1.-ss);let mut dd=vec![0.;p];for z in 0..p {dd[z]=scale*(dr[z]*(d.t[i]-off)-rate*doff[z]);}(tr,dd)};
 let mu=trend*factor+add;let r=d.y[i]-mu;lp+=-0.5*(r/sigma).powi(2)-sigma.ln();let w=r/(sigma*sigma);for z in 0..p{g[z]+=w*dt[z]*factor};for j in 0..d.kdim{g[b0+j]+=w*d.x[i*d.kdim+j]*(trend*d.sm[j]+d.sa[j])};g[2+s]+=r*r/(sigma*sigma)-1.; }
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=2+d.s+1+d.kdim||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
