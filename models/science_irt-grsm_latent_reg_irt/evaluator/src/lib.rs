use serde_json::Value;
use std::{ffi::{c_char,c_void},panic::{catch_unwind,AssertUnwindSafe},slice};

struct Data { i:usize,j:usize,n:usize,k:usize,m:usize, ii:Vec<usize>,jj:Vec<usize>,y:Vec<usize>, w_adj:Vec<f64> }
struct Bound { data:Data, ndim:usize }
struct Workspace;
fn nat(v:&Value,key:&str)->Result<usize,String>{let x=v.get(key).and_then(Value::as_u64).ok_or_else(||format!("{key} must be a positive integer"))? as usize;if x==0{Err(format!("{key} must be positive"))}else{Ok(x)}}
fn array(v:&Value,key:&str,n:usize)->Result<Vec<usize>,String>{let a=v.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be an array"))?;if a.len()!=n{return Err(format!("{key} length"))}a.iter().map(|x|x.as_u64().map(|z|z as usize).ok_or_else(||format!("{key} integer values"))).collect()}
fn parse_data(v:Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be JSON object")?; let vv=Value::Object(o.clone());
 let i=nat(&vv,"I")?;let j=nat(&vv,"J")?;let n=nat(&vv,"N")?;let k=nat(&vv,"K")?;
 let ii=array(&vv,"ii",n)?;let jj=array(&vv,"jj",n)?;let y=array(&vv,"y",n)?;
 if ii.iter().any(|&x|x<1||x>i)||jj.iter().any(|&x|x<1||x>j){return Err("response index bounds".into())}
 let m=*y.iter().max().ok_or("empty responses")?; if m<1{return Err("response maximum must be positive".into())}
 let wa=vv.get("W").and_then(Value::as_array).ok_or("W must be matrix")?;if wa.len()!=j{return Err("W rows".into())}
 let mut w=vec![0.;j*k];for r in 0..j {let row=wa[r].as_array().ok_or("W rows must be arrays")?;if row.len()!=k{return Err("W cols".into())}for c in 0..k{w[r*k+c]=row[c].as_f64().filter(|x|x.is_finite()).ok_or("W must be finite numeric")?}}
 // Exact transformed-data obtain_adjustments(), followed by W_adj construction.
 let mut adj_loc=vec![0.;k];let mut adj_scale=vec![1.;k];
 for c in 1..k {let col:Vec<f64>=(0..j).map(|r|w[r*k+c]).collect();let min=col.iter().copied().fold(f64::INFINITY,f64::min);let max=col.iter().copied().fold(f64::NEG_INFINITY,f64::max);let count=col.iter().filter(|&&x|x==min||x==max).count();let mean=col.iter().sum::<f64>()/(j as f64);adj_loc[c]=mean;adj_scale[c]=if count==j {max-min} else {(col.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/((j-1)as f64)).sqrt()*2.};if !adj_scale[c].is_finite()||adj_scale[c]==0. {return Err("transformed-data covariate scale invalid".into())}}
 for r in 0..j {for c in 0..k{w[r*k+c]=(w[r*k+c]-adj_loc[c])/adj_scale[c];}}
 Ok(Data{i,j,n,k,m,ii,jj,y,w_adj:w})
}
fn expected_layout(d:&Data)->String {let mut x=Vec::new();for z in 1..=d.i{x.push(format!("alpha.{z}"))}for z in 1..d.i{x.push(format!("beta_free.{z}"))}for z in 1..d.m{x.push(format!("kappa_free.{z}"))}for z in 1..=d.j{x.push(format!("theta.{z}"))}for z in 1..=d.k{x.push(format!("lambda_adj.{z}"))}x.join("\n")}
fn softplus(x:f64)->f64{if x>0. {x+(-x).exp().ln_1p()}else{x.exp().ln_1p()}}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 if q.iter().any(|x|!x.is_finite()){return Err("nonfinite position".into())};g.fill(0.);let i=d.i;let bf=i-1;let kf=d.m-1;let to=i+bf+kf;let lo=to+d.j;
 let mut lp=0.; let mut alpha=vec![0.;i];for z in 0..i {alpha[z]=q[z].exp(); if !alpha[z].is_finite(){return Err("alpha overflow".into())};lp-=0.5*(q[z]-1.).powi(2)+0.5*(2.*std::f64::consts::PI).ln();g[z]+=-(q[z]-1.);}
 let mut beta=vec![0.;i];for z in 0..bf{beta[z]=q[i+z]} beta[i-1]=-beta[..bf].iter().sum::<f64>();
 let norm3=3f64.ln()+0.5*(2.*std::f64::consts::PI).ln(); let mut gb=vec![0.;i];for z in 0..i{lp+=-0.5*(beta[z]/3.).powi(2)-norm3;gb[z]+=-beta[z]/9.;}
 let mut kap=vec![0.;d.m];for z in 0..kf{kap[z]=q[i+bf+z]}kap[d.m-1]=-kap[..kf].iter().sum::<f64>();let mut gk=vec![0.;d.m];for z in 0..d.m{lp+=-0.5*(kap[z]/3.).powi(2)-norm3;gk[z]+=-kap[z]/9.;}
 for person in 0..d.j {let mut mean=0.;for c in 0..d.k{mean+=d.w_adj[person*d.k+c]*q[lo+c]}let res=q[to+person]-mean;lp+=-0.5*res*res;g[to+person]-=res;for c in 0..d.k{g[lo+c]+=res*d.w_adj[person*d.k+c];}}
 for c in 0..d.k {let x=q[lo+c];lp+=-2.*(1.+x*x/3.).ln();g[lo+c]+=-4.*x/(3.+x*x);}
 for n in 0..d.n {let item=d.ii[n]-1;let person=d.jj[n]-1;let yy=d.y[n];let eta=q[to+person]*alpha[item]-beta[item];let mut util=vec![0.;d.m+1];for l in 1..=d.m{util[l]=(l as f64)*eta-kap[..l].iter().sum::<f64>();}let mx=util.iter().copied().fold(f64::NEG_INFINITY,f64::max);let es:Vec<f64>=util.iter().map(|x|(x-mx).exp()).collect();let sum=es.iter().sum::<f64>();lp+=util[yy]-mx-sum.ln();let el=es.iter().enumerate().map(|(l,p)|(l as f64)*p/sum).sum::<f64>();let de=yy as f64-el;g[to+person]+=alpha[item]*de;g[item]+=alpha[item]*q[to+person]*de;gb[item]-=de;for kk in 1..=d.m{let tail=es[kk..].iter().sum::<f64>()/sum;gk[kk-1]+=tail-if yy>=kk {1.}else{0.};}}
 for z in 0..bf{g[i+z]+=gb[z]-gb[i-1]}for z in 0..kf{g[i+bf+z]+=gk[z]-gk[d.m-1]}Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}fn fatal(p:*mut c_char,c:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,c,s.as_ref())};2}
#[no_mangle]pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=want.lines().count()||got!=want{return Err("dimension/layout mismatch".into())}Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gr:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||gr.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())}let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(gr,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
