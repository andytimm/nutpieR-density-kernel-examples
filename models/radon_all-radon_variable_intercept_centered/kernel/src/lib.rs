use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies of supplied Stan data. No data-only reductions are formed.
struct Data { j: usize, n: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x=finite_num(v,key)?;
    if x < 0.0 || x.fract()!=0.0 || x > usize::MAX as f64 { Err(format!("{key} must be a nonnegative integer")) } else { Ok(x as usize) }
}
fn num_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} has wrong length")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must contain finite numbers"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let vv=Value::Object(o.clone());
    let j=count(&vv,"J")?; let n=count(&vv,"N")?;
    if j==0 { return Err("J must be positive".into()); }
    let floor=num_array(o,"floor_measure",n)?;
    let y=num_array(o,"log_radon",n)?;
    let ca=o.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if ca.len()!=n { return Err("county_idx has wrong length".into()); }
    let mut county=Vec::with_capacity(n);
    for x in ca { let z=x.as_f64().filter(|z| z.is_finite() && *z>=1.0 && z.fract()==0.0 && *z<=j as f64).ok_or("county_idx must contain integers in 1:J")?; county.push(z as usize-1); }
    Ok(Data {j,n,county,floor,y})
}
fn expected_layout(d: &Data) -> String {
    let mut n: Vec<String>=(1..=d.j).map(|i| format!("alpha.{i}")).collect();
    n.extend(["beta".into(),"mu_alpha".into(),"sigma_alpha".into(),"sigma_y".into()]); n.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    // BridgeStan order: alpha.1...alpha.J, beta, mu_alpha, log(sigma_alpha), log(sigma_y).
    let ib=d.j; let im=ib+1; let isa=ib+2; let isy=ib+3;
    let beta=q[ib]; let mu=q[im]; let sa=q[isa].exp(); let sy=q[isy].exp();
    if !sa.is_finite() || !sy.is_finite() { return Err("non-finite scale transform".into()); }
    g.fill(0.0);
    // Sampling-statement priors are propto terms. Positive transforms add log-Jacobians.
    let mut lp=-0.5*sy*sy + q[isy] -0.5*sa*sa + q[isa] -0.5*(mu/10.0).powi(2) -0.5*(beta/10.0).powi(2);
    g[isy]=1.0-sy*sy; g[isa]=1.0-sa*sa; g[im]=-mu/100.0; g[ib]=-beta/100.0;
    let inv_sa2=1.0/(sa*sa); let mut sum_diff=0.0; let mut sum_diff2=0.0;
    for a in 0..d.j { let delta=q[a]-mu; sum_diff+=delta; sum_diff2+=delta*delta; g[a]-=delta*inv_sa2; }
    lp += -(d.j as f64)*sa.ln()-0.5*sum_diff2*inv_sa2;
    g[im] += sum_diff*inv_sa2;
    // Chain rule for sigma_alpha plus already-added log-Jacobian derivative.
    g[isa] += sa*(-(d.j as f64)/sa + sum_diff2/(sa*sa*sa));
    let inv_sy2=1.0/(sy*sy); let mut sum_res2=0.0;
    for n in 0..d.n { let a=d.county[n]; let res=d.y[n]-(q[a]+d.floor[n]*beta); sum_res2+=res*res; let z=res*inv_sy2; g[a]+=z; g[ib]+=d.floor[n]*z; }
    // Explicit normal_lpdf retains its normalizing constant under propto=true.
    lp += -(d.n as f64)*sy.ln()-0.5*sum_res2*inv_sy2-(d.n as f64)*0.5*(2.0*std::f64::consts::PI).ln();
    g[isy] += sy*(-(d.n as f64)/sy + sum_res2/(sy*sy*sy));
    Ok(lp)
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a [u8],String>{if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0 {let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 {unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.j+4||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast();};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w;};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=value;};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
