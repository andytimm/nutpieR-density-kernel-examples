use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

// Immutable copy of the original M-by-T capture histories.  We deliberately
// retain each observation rather than using any data-only count reduction.
struct Data { m: usize, t: usize, y: Vec<u8> }
struct Bound { data: Data, ndim: usize }
struct Workspace { p: Vec<f64> }

fn exact_usize(v: &Value, key: &str) -> Result<usize, String> {
    let n = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(n).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let m = exact_usize(&v, "M")?;
    let t = exact_usize(&v, "T")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != m { return Err("y row count does not match M".into()); }
    let mut y = Vec::with_capacity(m.checked_mul(t).ok_or("y dimensions overflow")?);
    for row in rows {
        let a = row.as_array().ok_or("y rows must be arrays")?;
        if a.len() != t { return Err("y column count does not match T".into()); }
        for x in a {
            match x.as_u64() { Some(0) => y.push(0), Some(1) => y.push(1), _ => return Err("y values must be 0 or 1".into()) }
        }
    }
    Ok(Data { m, t, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.t + 1);
    names.push("omega".to_string());
    for j in 1..=d.t { names.push(format!("p.{j}")); }
    names.join("\n")
}
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
#[inline] fn log_sigmoid(x: f64) -> f64 { if x >= 0.0 { -(-x).exp().ln_1p() } else { x - x.exp().ln_1p() } }
#[inline] fn log1m_sigmoid(x: f64) -> f64 { if x >= 0.0 { -x - (-x).exp().ln_1p() } else { -x.exp().ln_1p() } }

// Stan target, propto=true,jacobian=true. q is omega then p in BridgeStan order.
fn eval_model(d: &Data, w: &mut Workspace, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    let omega = sigmoid(q[0]);
    let mut log1mp_sum = 0.0;
    for j in 0..d.t { w.p[j] = sigmoid(q[j+1]); log1mp_sum += log1m_sigmoid(q[j+1]); }
    let r = log1mp_sum.exp();
    let mut lp = log_sigmoid(q[0]) + log1m_sigmoid(q[0]);
    g[0] = 1.0 - 2.0 * omega; // omega transform Jacobian
    for j in 0..d.t { lp += log_sigmoid(q[j+1]) + log1m_sigmoid(q[j+1]); g[j+1] = 1.0 - 2.0 * w.p[j]; }
    for i in 0..d.m {
        let row = &d.y[i*d.t..(i+1)*d.t];
        let observed = row.iter().any(|&x| x != 0);
        if observed {
            lp += log_sigmoid(q[0]);
            g[0] += 1.0 - omega;
            for j in 0..d.t {
                if row[j] == 1 { lp += log_sigmoid(q[j+1]); g[j+1] += 1.0-w.p[j]; }
                else { lp += log1m_sigmoid(q[j+1]); g[j+1] -= w.p[j]; }
            }
        } else {
            // log(1-omega + omega * product_j(1-p_j)); stable enough on finite q.
            let f = 1.0 - omega + omega * r;
            if !(f > 0.0) || !f.is_finite() { return Err("empty-history mixture domain".into()); }
            lp += f.ln();
            g[0] += omega * (1.0 - omega) * (r - 1.0) / f;
            for j in 0..d.t { g[j+1] -= omega * r * w.p[j] / f; }
        }
    }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) } else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) } }
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) { if !p.is_null() && cap != 0 { let n=s.len().min(cap-1); unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n); *p.add(n)=0; } } }
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p,cap,s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")}; unsafe{*out=std::ptr::null_mut()}; let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim != data.t+1 || got != want {return Err("dimension/layout mismatch".into())};Ok(Bound{data,ndim})}));match a {Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 {if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())};let b = unsafe { &*bound.cast::<Bound>() }; Ok(Box::into_raw(Box::new(Workspace { p: vec![0.0; b.data.t] })).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let qq=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let w=unsafe{&mut *workspace.cast::<Workspace>()};let value=eval_model(&b.data,w,qq,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)};unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
