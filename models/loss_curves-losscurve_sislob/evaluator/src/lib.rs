use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { growthmodel_id: i64, n_data: usize, n_time: usize, n_cohort: usize, cohort_id: Vec<usize>, t_idx: Vec<usize>, cohort_maxtime: Vec<usize>, t_value: Vec<f64>, premium: Vec<f64>, loss: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn finite_num(v: &Value, key: &str) -> Result<f64, String> { v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite()).ok_or_else(|| format!("{key} must be finite numeric")) }
fn integer(v:&Value,key:&str)->Result<i64,String>{ let x=finite_num(v,key)?; if x.fract()!=0. {Err(format!("{key} must be integer"))}else{Ok(x as i64)} }
fn arr_num(o:&serde_json::Map<String,Value>,key:&str,n:usize)->Result<Vec<f64>,String>{let a=o.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be array"))?;if a.len()!=n{return Err(format!("{key} length"))};a.iter().map(|x|x.as_f64().filter(|x|x.is_finite()).ok_or_else(||format!("{key} must be finite"))).collect()}
fn arr_int(o:&serde_json::Map<String,Value>,key:&str,n:usize,lo:i64,hi:i64)->Result<Vec<usize>,String>{let a=arr_num(o,key,n)?;a.into_iter().map(|x|if x.fract()!=0.||x <lo as f64||x>hi as f64{Err(format!("{key} bounds"))}else{Ok(x as usize)}).collect()}
fn parse_data(v:Value)->Result<Data,String>{let o=v.as_object().ok_or("data must be JSON object")?;let growthmodel_id=integer(&v,"growthmodel_id")?;if !(1..=2).contains(&growthmodel_id){return Err("growthmodel_id bounds".into())};let n_data=integer(&v,"n_data")? as usize;let n_time=integer(&v,"n_time")? as usize;let n_cohort=integer(&v,"n_cohort")? as usize;if n_data==0||n_time==0||n_cohort==0{return Err("dimensions must be positive".into())};let cohort_id=arr_int(o,"cohort_id",n_data,1,n_cohort as i64)?;let t_idx=arr_int(o,"t_idx",n_data,1,n_time as i64)?;let cohort_maxtime=arr_int(o,"cohort_maxtime",n_cohort,1,n_time as i64)?;let t_value=arr_num(o,"t_value",n_time)?;if t_value.iter().any(|x|*x<0.){return Err("t_value bounds".into())};let premium=arr_num(o,"premium",n_cohort)?;if premium.iter().any(|x|*x==0.) {return Err("premium must be nonzero".into())};let loss=arr_num(o,"loss",n_data)?;Ok(Data{growthmodel_id,n_data,n_time,n_cohort,cohort_id,t_idx,cohort_maxtime,t_value,premium,loss})}
fn expected_layout(d:&Data)->String{let mut z=vec!["omega".to_string(),"theta".to_string()];for i in 1..=d.n_cohort{z.push(format!("LR.{i}"))};z.push("mu_LR".into());z.push("sd_LR".into());z.push("loss_sd".into());z.join("\n")}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
 let omega=q[0].exp();let theta=q[1].exp();let lrs:Vec<f64>=q[2..2+d.n_cohort].iter().map(|x|x.exp()).collect();let mu=q[2+d.n_cohort];let sd=q[3+d.n_cohort].exp();let loss_sd=q[4+d.n_cohort].exp();
 let mut lp=0.;g.fill(0.);
 // lognormal priors after exp transforms; Jacobian cancels each -log constrained variable.
 lp+=-0.5*(mu/0.5).powi(2);g[2+d.n_cohort]+=-mu/0.25;
 const LOG_SQRT_2PI: f64 = 0.9189385332046727;
 lp+=-0.5*(q[3+d.n_cohort]/0.5).powi(2)-LOG_SQRT_2PI;g[3+d.n_cohort]+=-q[3+d.n_cohort]/0.25;
 lp+=-0.5*(q[4+d.n_cohort]/0.7).powi(2)-LOG_SQRT_2PI;g[4+d.n_cohort]+=-q[4+d.n_cohort]/0.49;
 lp+=-0.5*(q[0]/0.5).powi(2)-LOG_SQRT_2PI;g[0]+=-q[0]/0.25;
 lp+=-0.5*(q[1]/0.5).powi(2)-LOG_SQRT_2PI;g[1]+=-q[1]/0.25;
 for i in 0..d.n_cohort {let z=(q[2+i]-mu)/sd;lp+=-0.5*z*z-q[3+d.n_cohort]-LOG_SQRT_2PI;g[2+i]+=-z/sd;g[2+d.n_cohort]+=z/sd;g[3+d.n_cohort]+=z*z-1.;}
 for i in 0..d.n_data {let c=d.cohort_id[i]-1;let ti=d.t_idx[i]-1;let t=d.t_value[ti];let (gf,dgo,dgt)=if d.growthmodel_id==1 {let a=(t/theta).powf(omega);let e=(-a).exp();let f=1.-e;let da_o=omega*a*(t/theta).ln();let da_t=-omega*a; (f,e*da_o,e*da_t)} else {let r=(theta/t).powf(omega);let f=1./(1.+r);let v=f*(1.-f);(f,-omega*v*(theta/t).ln(),-omega*v)};let lm=lrs[c]*d.premium[c]*gf;let s=loss_sd*d.premium[c];let r=(d.loss[i]-lm)/s;lp+=-0.5*r*r-q[4+d.n_cohort]-d.premium[c].ln();let dl=r/s;g[2+c]+=dl*lm;g[0]+=dl*lrs[c]*d.premium[c]*dgo;g[1]+=dl*lrs[c]*d.premium[c]*dgt;g[4+d.n_cohort]+=r*r-1.;}
 Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 {
        let n = s.len().min(cap - 1);
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; }
    }
}
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 {
    unsafe { put_error(p, cap, s.as_ref()) }; 2
}

#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }

#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(
    json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char,
    layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize,
) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? })
            .map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?;
        let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? })
            .map_err(|_| "layout is not UTF-8")?;
        if ndim != 5 + data.n_cohort || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer {
        Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }
        Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density evaluator panic in bind"),
    }
}

#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() {
        unsafe { drop(Box::from_raw(bound.cast::<Bound>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(
    bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize,
) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        if bound.is_null() { return Err("null bound handle".into()); }
        Ok(Box::into_raw(Box::new(Workspace)).cast())
    }));
    match answer {
        Ok(Ok(w)) => { unsafe { *out = w; }; 0 }
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density evaluator panic in workspace"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() {
        unsafe { drop(Box::from_raw(w.cast::<Workspace>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(
    bound: *mut c_void, workspace: *mut c_void, q: *const f64, ndim: usize,
    lp: *mut f64, gradient: *mut f64, err: *mut c_char, cap: usize,
) -> i32 {
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if bound.is_null() || workspace.is_null() || lp.is_null() || gradient.is_null() {
            return Err("null evaluation handle/output".into());
        }
        let b = unsafe { &*bound.cast::<Bound>() };
        if ndim != b.ndim || (ndim != 0 && q.is_null()) { return Err("evaluation dimension".into()); }
        let q = if ndim == 0 { &[] } else { unsafe { slice::from_raw_parts(q, ndim) } };
        let g = unsafe { slice::from_raw_parts_mut(gradient, ndim) };
        let value = eval_model(&b.data, q, g)?;
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }
        unsafe { *lp = value; }; Ok(0)
    }));
    match answer {
        Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density evaluator panic in evaluate"),
    }
}