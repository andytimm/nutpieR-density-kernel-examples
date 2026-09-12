use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, j: usize, county: Vec<usize>, floor: Vec<f64>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn count(v: &Value, key: &str) -> Result<usize, String> {
    let x = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} too large"))
}
fn num_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let wrap = Value::Object(o.clone());
    let n = count(&wrap, "N")?; let j = count(&wrap, "J")?;
    if j == 0 { return Err("J must be positive".into()); }
    let floor = num_vec(o, "floor_measure", n)?; let y = num_vec(o, "log_radon", n)?;
    let a = o.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be integer array")?;
    if a.len() != n { return Err("county_idx length mismatch".into()); }
    let mut county = Vec::with_capacity(n);
    for x in a { let c = x.as_u64().ok_or("county_idx must be integer")?; if c == 0 || c > j as u64 { return Err("county_idx out of bounds".into()); } county.push(c as usize - 1); }
    Ok(Data { n, j, county, floor, y })
}
fn expected_layout(d: &Data) -> String {
    let mut x = vec!["sigma_y".to_owned(), "sigma_alpha".to_owned(), "sigma_beta".to_owned()];
    for i in 1..=d.j { x.push(format!("alpha_raw.{i}")); }
    for i in 1..=d.j { x.push(format!("beta_raw.{i}")); }
    x.push("mu_alpha".to_owned()); x.push("mu_beta".to_owned()); x.join("\n")
}
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let sy=q[0].exp(); let sa=q[1].exp(); let sb=q[2].exp();
    if !sy.is_finite() || !sa.is_finite() || !sb.is_finite() { return Err("scale transform overflow".into()); }
    let ao=3; let bo=3+d.j; let ma=q[3+2*d.j]; let mb=q[4+2*d.j];
    g.fill(0.0);
    let mut lp = -0.5*(sy*sy + sa*sa + sb*sb) + q[0]+q[1]+q[2]
        -0.5*(ma*ma + mb*mb)/100.0;
    let mut dsy = -sy;
    let mut dma = -ma/100.0; let mut dmb = -mb/100.0;
    for k in 0..d.j { lp += -0.5*q[ao+k]*q[ao+k] -0.5*q[bo+k]*q[bo+k]; g[ao+k] = -q[ao+k]; g[bo+k] = -q[bo+k]; }
    let inv_sy2=1.0/(sy*sy); let log_sqrt_2pi=0.91893853320467274178_f64;
    for n in 0..d.n {
        let k=d.county[n]; let alpha=ma+sa*q[ao+k]; let beta=mb+sb*q[bo+k];
        let r=d.y[n]-alpha-d.floor[n]*beta;
        lp += -0.5*r*r*inv_sy2 - sy.ln() - log_sqrt_2pi;
        let da=r*inv_sy2; let db=d.floor[n]*da;
        g[ao+k] += sa*da; g[bo+k] += sb*db;
        dma += da; dmb += db; dsy += r*r/(sy*sy*sy) - 1.0/sy;
    }
    let mut dsa=-sa; let mut dsb=-sb;
    for k in 0..d.j { dsa += (g[ao+k]+q[ao+k])*q[ao+k]/sa; dsb += (g[bo+k]+q[bo+k])*q[bo+k]/sb; }
    g[0]=dsy*sy+1.0; g[1]=dsa*sa+1.0; g[2]=dsb*sb+1.0; g[3+2*d.j]=dma; g[4+2*d.j]=dmb;
    Ok(lp)
}
unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())} else {Ok(unsafe {slice::from_raw_parts(p.cast(),n)})} }
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str) { if !p.is_null() && cap!=0 { let n=s.len().min(cap-1); unsafe {std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0;} } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe {put_error(p,cap,s.as_ref())}; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32 {1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?; let d=parse_data(v)?; let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != 3+2*d.j+2 || got != want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
