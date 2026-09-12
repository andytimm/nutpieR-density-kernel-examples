use serde_json::{Map, Value};
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copies are made at bind. `inter` is exactly the Stan transformed-data
// vector mom_hs .* mom_iq, rather than an added likelihood summary.
struct Data { n: usize, kid_score: Vec<f64>, mom_hs: Vec<f64>, mom_iq: Vec<f64>, inter: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(o: &Map<String, Value>, key: &str) -> Result<f64, String> {
    o.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn numeric_vec(o: &Map<String, Value>, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = o.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be a numeric array"))?;
    if a.len() != n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_f64().filter(|z| z.is_finite()).ok_or_else(|| format!("{key} must be finite numeric"))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let nf = finite_num(o, "N")?;
    if nf < 0.0 || nf.fract() != 0.0 || nf > usize::MAX as f64 { return Err("N must be a nonnegative integer".into()); }
    let n = nf as usize;
    let kid_score = numeric_vec(o, "kid_score", n)?;
    let mom_iq = numeric_vec(o, "mom_iq", n)?;
    let mom_hs = numeric_vec(o, "mom_hs", n)?;
    if kid_score.iter().any(|x| !(0.0..=200.0).contains(x)) { return Err("kid_score must be in [0, 200]".into()); }
    if mom_iq.iter().any(|x| !(0.0..=200.0).contains(x)) { return Err("mom_iq must be in [0, 200]".into()); }
    if mom_hs.iter().any(|x| !(0.0..=1.0).contains(x)) { return Err("mom_hs must be in [0, 1]".into()); }
    let inter = mom_hs.iter().zip(&mom_iq).map(|(h, iq)| h * iq).collect();
    Ok(Data { n, kid_score, mom_hs, mom_iq, inter })
}
fn expected_layout(_: &Data) -> String { "beta.1\nbeta.2\nbeta.3\nbeta.4\nsigma".into() }

fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let u = q[4];
    let sigma = u.exp();
    if !sigma.is_finite() || sigma == 0.0 { return Ok(f64::NAN); }
    let s2 = sigma * sigma;
    let inv_s2 = 1.0 / s2;
    // propto=true,jacobian=true: Cauchy retains -log1p((sigma/2.5)^2);
    // normal retains -log(sigma), while parameter-independent normalizers drop.
    let mut lp = u - (s2 / 6.25).ln_1p();
    g.fill(0.0);
    let mut gu = 1.0 - 2.0 * s2 / (6.25 + s2); // Jacobian plus Cauchy chain rule
    for i in 0..d.n {
        let eta = q[0] + q[1] * d.mom_hs[i] + q[2] * d.mom_iq[i] + q[3] * d.inter[i];
        let r = d.kid_score[i] - eta;
        let z2 = r * r * inv_s2;
        lp += -u - 0.5 * z2;
        let score = r * inv_s2;
        g[0] += score;
        g[1] += score * d.mom_hs[i];
        g[2] += score * d.mom_iq[i];
        g[3] += score * d.inter[i];
        gu += z2 - 1.0;
    }
    g[4] = gu;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {
 if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=5||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
