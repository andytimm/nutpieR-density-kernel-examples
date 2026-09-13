use serde_json::Value;
use std::{ffi::{c_char,c_void},panic::{catch_unwind,AssertUnwindSafe},slice};

struct Data { j:usize, n:usize, county:Vec<usize>, floor:Vec<f64>, y:Vec<f64> }
struct Bound { data:Data, ndim:usize }
struct Workspace;
fn number(v:&Value,k:&str)->Result<f64,String>{v.get(k).and_then(Value::as_f64).filter(|x|x.is_finite()).ok_or_else(||format!("{k} must be finite numeric"))}
fn count(v:&Value,k:&str,zero:bool)->Result<usize,String>{let x=number(v,k)?;if x.fract()!=0.||x<if zero {0.}else{1.}{Err(format!("{k} must be integer in range"))}else{Ok(x as usize)}}
fn numbers(v:&Value,k:&str,n:usize)->Result<Vec<f64>,String>{let a=v.get(k).and_then(Value::as_array).ok_or_else(||format!("{k} must be array"))?;if a.len()!=n{return Err(format!("{k} length"))};a.iter().map(|x|x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{k} must be finite"))).collect()}
fn parse_data(v:Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be JSON object")?; let vv=Value::Object(o.clone());
 let j=count(&vv,"J",true)?; if j==0{return Err("J must be positive".into())}; let n=count(&vv,"N",true)?;
 let floor=numbers(&vv,"floor_measure",n)?;let y=numbers(&vv,"log_radon",n)?;
 let a=vv.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be array")?;if a.len()!=n{return Err("county_idx length".into())};
 let mut county=Vec::with_capacity(n);for x in a {let z=x.as_f64().filter(|z|z.is_finite()&&z.fract()==0.&&*z>=1.&&*z<=j as f64).ok_or("county_idx range")?;county.push(z as usize-1)};
 Ok(Data{j,n,county,floor,y})
}
fn layout(d:&Data)->String {let mut x=Vec::with_capacity(d.j+4);x.push("alpha".into());for i in 1..=d.j{x.push(format!("beta_raw.{i}"))};x.push("mu_beta".into());x.push("sigma_beta".into());x.push("sigma_y".into());x.join("\n")}
fn eval(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 if q.iter().any(|x|!x.is_finite()){return Err("non-finite position".into())}; g.fill(0.);let j=d.j;let alpha=q[0];let mu=q[j+1];let sb=q[j+2].exp();let sy=q[j+3].exp();if !sb.is_finite()||!sy.is_finite(){return Err("transform overflow".into())};
 // Explicit target += normal_lpdf retains its normalizing constant under propto=true.
 let mut lp=-0.5*(alpha/10.).powi(2)-0.5*(mu/10.).powi(2)+q[j+2]-0.5*sb*sb+q[j+3]-0.5*sy*sy - (d.n as f64)*0.5*(2.0*std::f64::consts::PI).ln();
 g[0]=-alpha/100.;g[j+1]=-mu/100.;g[j+2]=1.-sb*sb;g[j+3]=1.-sy*sy;
 for i in 0..j {let b=q[1+i];lp-=0.5*b*b;g[1+i]=-b;}
 let invsy2=1./(sy*sy);
 for n in 0..d.n {let i=d.county[n];let raw=q[1+i];let eta=alpha+d.floor[n]*(mu+sb*raw);let r=d.y[n]-eta;let w=r*invsy2;lp+=-sy.ln()-0.5*r*w;g[0]+=w;g[1+i]+=d.floor[n]*sb*w;g[j+1]+=d.floor[n]*w;g[j+2]+=d.floor[n]*sb*raw*w;g[j+3]+=-1.+r*w;}
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle]pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,got:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let want=layout(&d);let actual=std::str::from_utf8(unsafe{bytes(got,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=d.j+4||actual!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle]pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let qq=unsafe{slice::from_raw_parts(q,ndim)};let gg=unsafe{slice::from_raw_parts_mut(g,ndim)};let x=eval(&b.data,qq,gg)?;if !x.is_finite()||gg.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
