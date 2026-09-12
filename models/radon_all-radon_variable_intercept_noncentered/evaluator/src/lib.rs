use serde_json::Value;
use std::{ffi::{c_char,c_void}, panic::{catch_unwind,AssertUnwindSafe}, slice};

struct Data { j: usize, n: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn usize_key(o: &serde_json::Map<String,Value>, key: &str) -> Result<usize,String> {
 let x=o.get(key).and_then(Value::as_u64).ok_or_else(||format!("{key} must be a nonnegative integer"))?;
 usize::try_from(x).map_err(|_|format!("{key} too large"))
}
fn num_array(o:&serde_json::Map<String,Value>,key:&str,n:usize)->Result<Vec<f64>,String>{
 let a=o.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be an array"))?;
 if a.len()!=n{return Err(format!("{key} length mismatch"));}
 a.iter().map(|x|x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v:Value)->Result<Data,String>{
 let o=v.as_object().ok_or("data must be a JSON object")?;
 let j=usize_key(o,"J")?; let n=usize_key(o,"N")?;
 if j==0{return Err("J must be positive".into());}
 let ca=o.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
 if ca.len()!=n{return Err("county_idx length mismatch".into());}
 let mut county=Vec::with_capacity(n);
 for x in ca {let k=x.as_u64().and_then(|z|usize::try_from(z).ok()).ok_or("county_idx must be integer")?; if k==0||k>j{return Err("county_idx out of bounds".into())};county.push(k-1);}
 Ok(Data{j,n,county,floor:num_array(o,"floor_measure",n)?,y:num_array(o,"log_radon",n)?})
}
fn expected_layout(d:&Data)->String { let mut x=(1..=d.j).map(|i|format!("alpha_raw.{i}")).collect::<Vec<_>>(); x.extend(["beta".into(),"mu_alpha".into(),"sigma_alpha".into(),"sigma_y".into()]); x.join("\n") }
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 let ib=d.j; let im=ib+1; let isa=ib+2; let isy=ib+3;
 let beta=q[ib];let m=q[im];let sa=q[isa].exp();let sy=q[isy].exp();
 if !sa.is_finite()||!sy.is_finite(){return Err("nonfinite scale transform".into())}
 g.fill(0.0); let mut lp=0.0;
 // Explicit normal_lpdf terms retain parameter-dependent scale terms under propto=true.
 for &a in &q[..d.j] {lp-=0.5*a*a;} lp-=0.5 * (2.0 * std::f64::consts::PI).ln() * d.n as f64; lp-=0.5*(beta/10.0).powi(2);lp-=0.5*(m/10.0).powi(2);lp += -0.5*sa*sa + q[isa]; lp += -0.5*sy*sy + q[isy];
 g[ib]=-beta/100.0;g[im]=-m/100.0;g[isa]=1.0-sa*sa;g[isy]=1.0-sy*sy;
 let invsy2=1.0/(sy*sy);let mut dsa=0.0;let mut dsy=0.0;
 for n in 0..d.n {let i=d.county[n];let a=q[i];let r=d.y[n]-(m+sa*a+d.floor[n]*beta); lp-=0.5*r*r*invsy2+sy.ln();let z=r*invsy2;g[i]+=z*sa;g[ib]+=z*d.floor[n];g[im]+=z;dsa+=z*a;dsy+=r*r/(sy*sy*sy)-1.0/sy;}
 for i in 0..d.j {g[i]-=q[i];} g[isa]+=dsa*sa;g[isy]+=dsy*sy;
 Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let d=parse_data(serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=d.j+4||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into())};let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
