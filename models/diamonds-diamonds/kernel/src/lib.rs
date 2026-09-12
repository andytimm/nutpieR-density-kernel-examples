use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe}, slice};

struct Data { n: usize, kc: usize, y: Vec<f64>, xc: Vec<f64>, prior_only: bool }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn number(v:&Value, name:&str)->Result<f64,String>{v.as_f64().filter(|x|x.is_finite()).ok_or_else(||format!("{name} must be finite numeric"))}
fn integer(v:&Value,name:&str)->Result<usize,String>{let x=number(v,name)?;if x<0.||x.fract()!=0. {Err(format!("{name} must be nonnegative integer"))}else{Ok(x as usize)}}
fn parse_data(v:Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be a JSON object")?;
 let n=integer(o.get("N").ok_or("missing N")?,"N")?; if n==0{return Err("N must be positive".into())}
 let k=integer(o.get("K").ok_or("missing K")?,"K")?; if k<2{return Err("K must be at least 2".into())}; let kc=k-1;
 let prior=integer(o.get("prior_only").ok_or("missing prior_only")?,"prior_only")?; if prior>1{return Err("prior_only must be 0 or 1".into())}
 let ya=o.get("Y").and_then(Value::as_array).ok_or("Y must be array")?;if ya.len()!=n{return Err("Y length mismatch".into())};let mut y=Vec::with_capacity(n);for z in ya {y.push(number(z,"Y")?)}
 let xa=o.get("X").and_then(Value::as_array).ok_or("X must be array")?;if xa.len()!=n{return Err("X row count mismatch".into())}
 // This exactly reproduces Stan transformed data: means for columns 2:K then Xc = X[,2:K]-means.
 let mut means=vec![0.;kc]; let mut xfull=Vec::with_capacity(n*k);
 for row in xa {let a=row.as_array().ok_or("X rows must be arrays")?;if a.len()!=k{return Err("X column count mismatch".into())};for (j,z) in a.iter().enumerate(){let x=number(z,"X")?;if j>0 {means[j-1]+=x;}xfull.push(x)}}
 for x in &mut means {*x/=n as f64};let mut xc=Vec::with_capacity(n*kc);for i in 0..n {for j in 0..kc {xc.push(xfull[i*k+j+1]-means[j])}}
 Ok(Data{n,kc,y,xc,prior_only:prior==1})
}
fn expected_layout(d:&Data)->String{let mut x=(1..=d.kc).map(|j|format!("b.{j}")).collect::<Vec<_>>();x.push("Intercept".into());x.push("sigma".into());x.join("\n")}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 let k=d.kc; let intercept=q[k]; let ls=q[k+1]; let sigma=ls.exp(); if !sigma.is_finite()||sigma<=0. {return Err("nonfinite sigma transform".into())}; let invs=1./sigma; let invs2=invs*invs;
 for z in g.iter_mut(){*z=0.}; let mut lp=0.;
 // normal_lpdf(b | 0,1), propto=true: unit scale makes its retained term -b^2/2.
 for j in 0..k {lp-=0.5*q[j]*q[j] + 0.5*(2.0*std::f64::consts::PI).ln();g[j]-=q[j];}
 // student_t_lpdf(Intercept | 3,8,10), with constant terms omitted by propto=true.
 let tconst=2_f64.ln()-0.5*3_f64.ln()-std::f64::consts::PI.ln()-10_f64.ln(); let z=(intercept-8.)/10.; let a=1.+z*z/3.; lp+=tconst-2.*a.ln();g[k]-=4.*z/(30.*a);
 // student_t_lpdf(sigma | 3,0,10) - student_t_lccdf(0|3,0,10); lccdf is constant.
 let zs=sigma/10.;let as_=1.+zs*zs/3.;lp+=tconst+2_f64.ln()-2.*as_.ln();g[k+1]+=-4.*zs*sigma/(30.*as_);
 // Jacobian of sigma=exp(ls).
 lp+=ls;g[k+1]+=1.;
 if !d.prior_only {for i in 0..d.n {let row=&d.xc[i*k..(i+1)*k];let mut eta=intercept;for j in 0..k{eta+=row[j]*q[j]};let r=d.y[i]-eta;lp-=0.5*r*r*invs2+ls+0.5*(2.0*std::f64::consts::PI).ln();let c=r*invs2;for j in 0..k{g[j]+=row[j]*c};g[k]+=c;g[k+1]+=r*r*invs2-1.;}}
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,c:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,c,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut*mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=d.kc+2||got!=want{return Err(format!("dimension/layout mismatch: ndim={ndim} want={}, got={got:?}, want={want:?}",d.kc+2))};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut*mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|z|!z.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
