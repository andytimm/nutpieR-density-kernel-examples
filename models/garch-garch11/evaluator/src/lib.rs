use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { y: Vec<f64>, sigma1: f64 }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let t = o.get("T").and_then(Value::as_u64).ok_or("T must be a nonnegative integer")? as usize;
    let yv = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if yv.len() != t { return Err("T must equal length(y)".into()); }
    let mut y = Vec::with_capacity(t);
    for x in yv { y.push(x.as_f64().filter(|z| z.is_finite()).ok_or("y must be finite numeric")?); }
    let sigma1 = finite_num(&Value::Object(o.clone()), "sigma1")?;
    if sigma1 <= 0.0 { return Err("sigma1 must be > 0".into()); }
    Ok(Data { y, sigma1 })
}
fn expected_layout(_: &Data) -> String { "mu\nalpha0\nalpha1\nbeta1".into() }
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let z=x.exp(); z/(1.0+z) } }
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let mu=q[0]; let alpha0=q[1].exp(); let alpha1=sigmoid(q[2]); let sbet=sigmoid(q[3]);
    let beta1=(1.0-alpha1)*sbet;
    if !mu.is_finite() || !alpha0.is_finite() || !beta1.is_finite() { return Err("non-finite transform".into()); }
    // Accumulate the likelihood and its constrained-coordinate derivatives in one pass.
    let mut lp=0.0; let mut gc=[0.0;4];
    let mut sigma=d.sigma1; let mut ds=[0.0;4];
    for (t, &y) in d.y.iter().enumerate() {
        let z=(y-mu)/sigma;
        lp += -0.5*z*z - sigma.ln();
        let factor=(z*z-1.0)/sigma;
        gc[0] += z/sigma + factor*ds[0];
        gc[1] += factor*ds[1]; gc[2] += factor*ds[2]; gc[3] += factor*ds[3];
        if t+1 < d.y.len() {
            let r=y-mu; let old_sigma=sigma; let old_ds=ds;
            let v=alpha0 + alpha1*r*r + beta1*old_sigma*old_sigma;
            sigma=v.sqrt();
            if !sigma.is_finite() || sigma <= 0.0 { return Err("invalid volatility recursion".into()); }
            for j in 0..4 {
                let dmu=if j==0 {1.0} else {0.0};
                let da0=if j==1 {1.0} else {0.0};
                let da1=if j==2 {1.0} else {0.0};
                let db1=if j==3 {1.0} else {0.0};
                ds[j]=(da0 + da1*r*r - 2.0*alpha1*r*dmu + db1*old_sigma*old_sigma
                         + 2.0*beta1*old_sigma*old_ds[j])/(2.0*sigma);
            }
        }
    }
    // Stan unconstraining transforms, including both dependent upper-bound Jacobians.
    lp += q[1] + alpha1.ln() + (1.0-alpha1).ln() + (1.0-alpha1).ln()
        + sbet.ln() + (1.0-sbet).ln();
    g[0]=gc[0];
    g[1]=gc[1]*alpha0 + 1.0;
    g[2]=gc[2]*alpha1*(1.0-alpha1) - gc[3]*alpha1*beta1 + 1.0 - 3.0*alpha1;
    g[3]=gc[3]*(1.0-alpha1)*sbet*(1.0-sbet) + 1.0 - 2.0*sbet;
    Ok(lp)
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=4||got!=want{return Err("dimension/layout mismatch".into())}Ok(Bound{data,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut()}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
