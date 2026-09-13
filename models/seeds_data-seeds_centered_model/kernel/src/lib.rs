use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: Vec<usize>, total: Vec<usize>, x1: Vec<f64>, x2: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn integer(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.as_f64().filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0 && *x <= usize::MAX as f64)
        .ok_or_else(|| format!("{key} must contain nonnegative integer values"))?;
    Ok(x as usize)
}
fn numbers(v: &Value, key: &str, len: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != len { return Err(format!("{key} has wrong length")); }
    a.iter().map(|z| z.as_f64().filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must contain finite numeric values"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let i = integer(o.get("I").ok_or("missing I")?, "I")?;
    let n_v = o.get("n").and_then(Value::as_array).ok_or("n must be an array")?;
    let total_v = o.get("N").and_then(Value::as_array).ok_or("N must be an array")?;
    if n_v.len() != i || total_v.len() != i { return Err("n or N has wrong length".into()); }
    let mut n = Vec::with_capacity(i); let mut total = Vec::with_capacity(i);
    for j in 0..i { let nj=integer(&n_v[j], "n")?; let n_total=integer(&total_v[j], "N")?; if nj > n_total { return Err("n must be <= N".into()); } n.push(nj); total.push(n_total); }
    Ok(Data { n, total, x1: numbers(&Value::Object(o.clone()), "x1", i)?, x2: numbers(&Value::Object(o.clone()), "x2", i)? })
}
fn expected_layout(d: &Data) -> String {
    let mut names = vec!["alpha0".to_string(), "alpha1".to_string(), "alpha12".to_string(), "alpha2".to_string()];
    names.extend((1..=d.n.len()).map(|j| format!("c.{j}")));
    names.push("sigma".to_string()); names.join("\n")
}
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let i = d.n.len(); let a0=q[0]; let a1=q[1]; let a12=q[2]; let a2=q[3];
    let c=&q[4..4+i]; let log_sigma=q[4+i]; let sigma=log_sigma.exp();
    if !sigma.is_finite() { return Err("non-finite sigma transform".into()); }
    g.fill(0.0);
    g[0] = -a0; g[1] = -a1; g[2] = -a12; g[3] = -a2;
    let mut lp = -0.5*(a0*a0+a1*a1+a12*a12+a2*a2);
    // Original c ~ normal(0, sigma), retaining its parameter-dependent scale term.
    let inv_sigma2=1.0/(sigma*sigma); let mut c_sq=0.0;
    for j in 0..i { c_sq += c[j]*c[j]; g[4+j] -= c[j]*inv_sigma2; }
    lp += -0.5*c_sq*inv_sigma2 - (i as f64)*log_sigma;
    // sigma ~ cauchy(0, 1), followed by the lower-bound exp transform Jacobian.
    let sigma2=sigma*sigma; lp += -(1.0+sigma2).ln() + log_sigma;
    g[4+i] = c_sq*inv_sigma2 - (i as f64) - 2.0*sigma2/(1.0+sigma2) + 1.0;
    // Original transformed data x1x2 is computed here; no data-only summary is formed at bind.
    let mean_c=c.iter().sum::<f64>()/(i as f64); let mut sum_score=0.0;
    for j in 0..i {
        let eta=a0+a1*d.x1[j]+a2*d.x2[j]+a12*(d.x1[j]*d.x2[j])+(c[j]-mean_c);
        lp += (d.n[j] as f64)*eta - (d.total[j] as f64)*softplus(eta);
        let score=(d.n[j] as f64)-(d.total[j] as f64)*sigmoid(eta);
        g[0]+=score; g[1]+=score*d.x1[j]; g[2]+=score*d.x1[j]*d.x2[j]; g[3]+=score*d.x2[j]; g[4+j]+=score; sum_score+=score;
    }
    let avg_score=sum_score/(i as f64); for j in 0..i { g[4+j]-=avg_score; }
    Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a [u8],String>{if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.n.len()+5||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast();};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match answer{Ok(Ok(w))=>{unsafe{*out=w;};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let answer=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=value;};Ok(0)}));match answer{Ok(Ok(status))=>status,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
