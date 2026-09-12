use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, x: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace { k: Vec<f64>, l: Vec<f64>, inv: Vec<f64>, v: Vec<f64> }

fn num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn int(v: &Value, key: &str) -> Result<usize, String> {
    let x=num(v,key)?; if x < 1.0 || x.fract()!=0.0 { return Err(format!("{key} must be positive integer")); } Ok(x as usize)
}
fn vec(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a=v.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|z|z.as_f64().filter(|x|x.is_finite()).ok_or_else(||format!("{key} must be finite"))).collect()
}
fn parse_data(v: Value) -> Result<Data,String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let v=Value::Object(o.clone()); let n=int(&v,"N")?;
    Ok(Data {n, x:vec(&v,"x",n)?, y:vec(&v,"y",n)?})
}
fn expected_layout(_: &Data)->String { "rho\nalpha\nsigma".into() }

fn chol(a: &[f64], l: &mut [f64], n:usize) -> Result<(),String> {
    l.fill(0.0);
    for i in 0..n { for j in 0..=i {
        let mut s=a[i*n+j]; for k in 0..j { s-=l[i*n+k]*l[j*n+k]; }
        if i==j { if !(s>0.0) || !s.is_finite() {return Err("covariance is not positive definite".into());} l[i*n+j]=s.sqrt(); }
        else { l[i*n+j]=s/l[j*n+j]; }
    }} Ok(())
}
fn eval_model(d:&Data, q:&[f64], g:&mut[f64], w:&mut Workspace)->Result<f64,String> {
    let n=d.n; let rho=q[0].exp(); let alpha=q[1].exp(); let sigma=q[2].exp();
    if !rho.is_finite() || !alpha.is_finite() || !sigma.is_finite() { return Err("non-finite transformed parameter".into()); }
    let a2=alpha*alpha; w.k.fill(0.0);
    for i in 0..n { for j in 0..=i { let dx=d.x[i]-d.x[j]; let z=-(0.5*dx*dx/(rho*rho)); let v=a2*z.exp() + if i==j {sigma} else {0.0}; w.k[i*n+j]=v; w.k[j*n+i]=v; }}
    chol(&w.k,&mut w.l,n)?;
    // v = K^-1 y via triangular solves
    for i in 0..n { let mut s=d.y[i]; for j in 0..i{s-=w.l[i*n+j]*w.v[j];} w.v[i]=s/w.l[i*n+i]; }
    for ii in 0..n { let i=n-1-ii; let mut s=w.v[i]; for j in i+1..n{s-=w.l[j*n+i]*w.v[j];} w.v[i]=s/w.l[i*n+i]; }
    // explicit inverse via one triangular solve per column; small original N=11 covariance
    w.inv.fill(0.0);
    for col in 0..n { for i in 0..n { let mut s=if i==col {1.0}else{0.0}; for j in 0..i{s-=w.l[i*n+j]*w.inv[j*n+col];} w.inv[i*n+col]=s/w.l[i*n+i]; }
        for ii in 0..n {let i=n-1-ii; let mut s=w.inv[i*n+col]; for j in i+1..n{s-=w.l[j*n+i]*w.inv[j*n+col];} w.inv[i*n+col]=s/w.l[i*n+i];} }
    let mut logdet=0.0; for i in 0..n {logdet+=w.l[i*n+i].ln();}
    let quad: f64=d.y.iter().zip(w.v.iter()).map(|(y,v)|y*v).sum();
    let mut lp= -0.5*quad-logdet + 25.0*q[0]-4.0*rho -0.5*a2/4.0+q[1] -0.5*sigma*sigma+q[2];
    let mut gr=25.0-4.0*rho; let mut ga=1.0-a2/4.0; let mut gs=1.0-sigma*sigma;
    // d log MVN / dK_ij = .5*(v_i v_j - K^-1_ij); sum all entries, including off diagonal.
    for i in 0..n { for j in 0..n { let h=0.5*(w.v[i]*w.v[j]-w.inv[i*n+j]); let dx=d.x[i]-d.x[j]; let base=a2*(-0.5*dx*dx/(rho*rho)).exp(); gr += h*base*(dx*dx/(rho*rho)); ga += h*(2.0*base); if i==j {gs += h*sigma;} }}
    g[0]=gr;g[1]=ga;g[2]=gs;
    if !lp.is_finite(){return Err("non-finite log density".into());} Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=3||got!=expected_layout(&d){return Err("dimension/layout mismatch".into());}Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into());}let n=unsafe{(&*b.cast::<Bound>()).data.n};Ok(Box::into_raw(Box::new(Workspace{k:vec![0.;n*n],l:vec![0.;n*n],inv:vec![0.;n*n],v:vec![0.;n]})).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||q.is_null(){return Err("evaluation dimension".into());}let q=unsafe{slice::from_raw_parts(q,ndim)};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let w=unsafe{&mut *w.cast::<Workspace>()};let v=eval_model(&b.data,q,g,w)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
