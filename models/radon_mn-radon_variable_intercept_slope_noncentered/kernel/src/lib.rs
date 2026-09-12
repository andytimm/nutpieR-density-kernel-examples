use serde_json::Value;
use std::{ffi::{c_char,c_void},panic::{catch_unwind,AssertUnwindSafe},slice};
struct Data { n: usize, j: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn num(v:&Value, name:&str)->Result<f64,String>{v.as_f64().filter(|x|x.is_finite()).ok_or_else(||format!("{name} must contain finite numbers"))}
fn vec_num(o:&serde_json::Map<String,Value>, name:&str)->Result<Vec<f64>,String>{o.get(name).and_then(Value::as_array).ok_or_else(||format!("{name} must be array"))?.iter().map(|x|num(x,name)).collect()}
fn parse_data(v:Value)->Result<Data,String>{ let o=v.as_object().ok_or("data must be JSON object")?;
 let n=o.get("N").and_then(Value::as_u64).ok_or("N must be nonnegative integer")? as usize;
 let j=o.get("J").and_then(Value::as_u64).filter(|&x|x>0).ok_or("J must be positive integer")? as usize;
 let county=o.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be array")?.iter().map(|x| x.as_u64().map(|z|z as usize).filter(|&z|z>=1&&z<=j).ok_or("county_idx out of bounds")).collect::<Result<Vec<_>,_>>()?;
 let floor=vec_num(o,"floor_measure")?; let y=vec_num(o,"log_radon")?;
 if county.len()!=n||floor.len()!=n||y.len()!=n {return Err("data lengths must equal N".into())}; Ok(Data{n,j,county,floor,y}) }
fn expected_layout(d:&Data)->String { let mut x=vec!["sigma_y".to_string(),"sigma_alpha".to_string(),"sigma_beta".to_string()]; for p in ["alpha_raw","beta_raw"] {for i in 1..=d.j{x.push(format!("{p}.{i}"));}} x.push("mu_alpha".into());x.push("mu_beta".into());x.join("\n") }
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 let sy=q[0].exp();let sa=q[1].exp();let sb=q[2].exp(); if !sy.is_finite()||!sa.is_finite()||!sb.is_finite(){return Err("invalid scale transform".into())};
 for z in g.iter_mut(){*z=0.0}; let ao=3;let bo=3+d.j;let ma=3+2*d.j;let mb=ma+1;
 // Propto Stan density: normal constants are omitted; positive transforms add q[0..3].
 let mut lp=q[0]+q[1]+q[2] -0.5*(sy*sy+sa*sa+sb*sb) -0.5*(q[ma]*q[ma]+q[mb]*q[mb])/100.0 - 0.5*(d.n as f64)*(2.0*std::f64::consts::PI).ln();
 g[0]=1.0-sy*sy;g[1]=1.0-sa*sa;g[2]=1.0-sb*sb;g[ma]=-q[ma]/100.0;g[mb]=-q[mb]/100.0;
 for k in 0..d.j {let a=q[ao+k];let b=q[bo+k];lp-=0.5*(a*a+b*b);g[ao+k]-=a;g[bo+k]-=b;}
 for n in 0..d.n {let k=d.county[n]-1;let eta=q[ma]+sa*q[ao+k]+d.floor[n]*(q[mb]+sb*q[bo+k]);let r=d.y[n]-eta;let inv=1.0/(sy*sy);lp+=-q[0]-0.5*r*r*inv; let de=r*inv; g[ao+k]+=de*sa;g[bo+k]+=de*d.floor[n]*sb;g[ma]+=de;g[mb]+=de*d.floor[n];g[1]+=de*sa*q[ao+k];g[2]+=de*d.floor[n]*sb*q[bo+k];g[0]+=-1.0+r*r*inv;}
 Ok(lp) }
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let data=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=3+2*data.j+2||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|z|!z.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
