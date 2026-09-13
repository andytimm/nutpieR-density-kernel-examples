use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, u: Vec<f64>, v: Vec<f64>, alpha: [[f64; 2]; 2] }
struct Bound { data: Data, ndim: usize }
struct Workspace { gamma: Vec<[f64; 2]>, deriv: Vec<[[f64; 8]; 2]> }

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))
}
fn number_array(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z|z.is_finite()).ok_or_else(||format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let k=finite_num(&Value::Object(o.clone()),"K")?;
    let n=finite_num(&Value::Object(o.clone()),"N")?;
    if k != 2.0 || n < 1.0 || n.fract() != 0.0 { return Err("this Stan program requires K=2 and positive integer N".into()); }
    let n=n as usize;
    let u=number_array(o,"u",n)?; let vv=number_array(o,"v",n)?;
    if u.iter().any(|x|*x<0.0)||vv.iter().any(|x|*x<0.0) { return Err("exponential observations must be nonnegative".into()); }
    let a=o.get("alpha").and_then(Value::as_array).ok_or("alpha must be a matrix")?;
    if a.len()!=2 { return Err("alpha shape mismatch".into()); }
    let mut alpha=[[0.;2];2];
    for i in 0..2 { let r=a[i].as_array().ok_or("alpha must be a matrix")?; if r.len()!=2{return Err("alpha shape mismatch".into())}; for j in 0..2 { alpha[i][j]=r[j].as_f64().filter(|x|x.is_finite()&&*x>=0.).ok_or("alpha must be finite and nonnegative")?; } }
    Ok(Data {n,u,v:vv,alpha})
}
fn expected_layout(_: &Data) -> String { "theta1.1\ntheta2.1\nphi.1\nphi.2\nlambda.1\nlambda.2".into() }
fn log_gamma(z: f64) -> f64 {
    // Lanczos approximation; alpha is positive and this preserves explicit dirichlet constants.
    const C: [f64; 9] = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if z < 0.5 { return std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - log_gamma(1.0-z); }
    let x=z-1.0; let mut a=C[0]; for i in 1..C.len() { a += C[i]/(x+i as f64); }
    let t=x+7.5; 0.5*(2.0*std::f64::consts::PI).ln()+(x+0.5)*t.ln()-t+a.ln()
}
fn sigmoid(x:f64)->f64 { if x>=0. {1./(1.+(-x).exp())} else {let z=x.exp();z/(1.+z)} }
fn lse2(a:f64,b:f64)->f64 { let m=a.max(b); m + ((a-m).exp() + (b-m).exp()).ln() }
fn eval_model(d:&Data,q:&[f64],g:&mut[f64],w:&mut Workspace)->Result<f64,String> {
    if q.iter().any(|x|!x.is_finite()) { return Err("non-finite position".into()); }
    // Stan 2.37+ simplex uses the ILR transform.  For K=2 this is
    // theta[1] = inv_logit(sqrt(2) * q), theta[2] = 1 - theta[1].
    let scale = 2.0_f64.sqrt();
    let s=sigmoid(scale*q[0]); let t=sigmoid(scale*q[1]);
    let th=[[s,1.-s],[t,1.-t]];
    let p0=q[2].exp(); let p1=p0+q[3].exp(); let l0=q[4].exp(); let l1=l0+q[5].exp();
    if !p0.is_finite()||!p1.is_finite()||!l0.is_finite()||!l1.is_finite(){return Err("transform overflow".into())}
    let ph=[p0,p1]; let la=[l0,l1];
    let mut gt=[[0.;2];2]; let mut gp=[0.;2]; let mut gl=[0.;2];
    // Explicit Stan *_lpdf calls retain normalizing terms even under propto=true.
    let mut lp=0.;
    for r in 0..2 { let asum=d.alpha[r][0]+d.alpha[r][1]; lp+=log_gamma(asum)-log_gamma(d.alpha[r][0])-log_gamma(d.alpha[r][1]); for c in 0..2 { lp+=(d.alpha[r][c]-1.)*th[r][c].ln(); gt[r][c]+=(d.alpha[r][c]-1.)/th[r][c]; } }
    let log2pi=(2.*std::f64::consts::PI).ln();
    for k in 0..2 { let m=if k==0 {0.}else{3.}; lp+=-0.5*(ph[k]-m)*(ph[k]-m)-0.5*log2pi; gp[k]-=ph[k]-m; lp+=-0.5*(la[k]-m)*(la[k]-m)-0.5*log2pi; gl[k]-=la[k]-m; }
    for k in 0..2 { let e=ph[k].ln()-ph[k]*d.u[0]+la[k].ln()-la[k]*d.v[0]; w.gamma[0][k]=e; w.deriv[0][k]=[0.;8]; w.deriv[0][k][4+k]=1./ph[k]-d.u[0]; w.deriv[0][k][6+k]=1./la[k]-d.v[0]; }
    for i in 1..d.n { for k in 0..2 { let a=w.gamma[i-1][0]+th[0][k].ln(); let b=w.gamma[i-1][1]+th[1][k].ln(); let z=lse2(a,b); let wa=(a-z).exp(); let wb=(b-z).exp(); let e=ph[k].ln()-ph[k]*d.u[i]+la[k].ln()-la[k]*d.v[i]; w.gamma[i][k]=z+e; for h in 0..8 { w.deriv[i][k][h]=wa*w.deriv[i-1][0][h]+wb*w.deriv[i-1][1][h]; } w.deriv[i][k][2*0+k]+=wa/th[0][k]; w.deriv[i][k][2*1+k]+=wb/th[1][k]; w.deriv[i][k][4+k]+=1./ph[k]-d.u[i]; w.deriv[i][k][6+k]+=1./la[k]-d.v[i]; } }
    let z=lse2(w.gamma[d.n-1][0],w.gamma[d.n-1][1]); lp+=z; let wa=(w.gamma[d.n-1][0]-z).exp(); let wb=(w.gamma[d.n-1][1]-z).exp(); let mut h=[0.;8]; for j in 0..8 {h[j]=wa*w.deriv[d.n-1][0][j]+wb*w.deriv[d.n-1][1][j];}
    for r in 0..2 {for c in 0..2 {gt[r][c]+=h[2*r+c];}} for k in 0..2 {gp[k]+=h[4+k];gl[k]+=h[6+k];}
    // Stan Math simplex_constrain(y, lp): sum(log(theta)) + 0.5*log(K).
    // K=2 and there are two simplex rows, so retain log(2) in log density.
    // The constant has zero derivative; the ILR coordinate scale remains below.
    lp += 2.0_f64.ln();
    g[0]=scale*((gt[0][0]-gt[0][1])*s*(1.-s)+(1.-2.*s));
    g[1]=scale*((gt[1][0]-gt[1][1])*t*(1.-t)+(1.-2.*t));
    g[2]=(gp[0]+gp[1])*p0+1.; g[3]=gp[1]*(p1-p0)+1.; g[4]=(gl[0]+gl[1])*l0+1.; g[5]=gl[1]*(l1-l0)+1.;
    lp+=s.ln()+(1.-s).ln()+t.ln()+(1.-t).ln()+q[2]+q[3]+q[4]+q[5]; Ok(lp)
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=6||got!=expected_layout(&d){return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};let n=unsafe { (&*b.cast::<Bound>()).data.n }; Ok(Box::into_raw(Box::new(Workspace{gamma:vec![[0.;2];n],deriv:vec![[[0.;8];2];n]})).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let bb=unsafe{&*b.cast::<Bound>()};if ndim!=bb.ndim||q.is_null(){return Err("evaluation dimension".into())};let x=unsafe{slice::from_raw_parts(q,ndim)};let gg=unsafe{slice::from_raw_parts_mut(g,ndim)};let ww=unsafe{&mut*w.cast::<Workspace>()};let val=eval_model(&bb.data,x,gg,ww)?;if !val.is_finite()||gg.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=val};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
