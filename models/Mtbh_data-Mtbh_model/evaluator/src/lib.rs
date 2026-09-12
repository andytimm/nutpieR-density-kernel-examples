use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { m: usize, t: usize, y: Vec<u8>, observed: Vec<bool> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn parse_usize(v: &Value, key: &str) -> Result<usize, String> {
    let n = v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a nonnegative integer"))?;
    usize::try_from(n).map_err(|_| format!("{key} is too large"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let m = parse_usize(&Value::Object(o.clone()), "M")?;
    let t = parse_usize(&Value::Object(o.clone()), "T")?;
    let rows = o.get("y").and_then(Value::as_array).ok_or("y must be an array")?;
    if rows.len() != m { return Err("y row count does not match M".into()); }
    let mut y = Vec::with_capacity(m.checked_mul(t).ok_or("data dimensions overflow")?);
    let mut observed = Vec::with_capacity(m);
    for row in rows {
        let a = row.as_array().ok_or("y rows must be arrays")?;
        if a.len() != t { return Err("y column count does not match T".into()); }
        let mut any = false;
        for x in a {
            let z = x.as_u64().filter(|z| *z <= 1).ok_or("y must contain integer 0 or 1")? as u8;
            any |= z == 1; y.push(z);
        }
        observed.push(any); // exactly the Stan transformed-data s[i] > 0 test
    }
    Ok(Data { m, t, y, observed })
}
fn expected_layout(d: &Data) -> String {
    let mut names = Vec::with_capacity(d.t + d.m + 3);
    names.push("omega".to_string());
    for j in 1..=d.t { names.push(format!("mean_p.{j}")); }
    names.push("gamma".to_string()); names.push("sigma".to_string());
    for i in 1..=d.m { names.push(format!("eps_raw.{i}")); }
    names.join("\n")
}
fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
fn log_sigmoid(x: f64) -> f64 { -softplus(-x) }
fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
fn logsumexp(a: f64, b: f64) -> f64 { let z=a.max(b); z + ((a-z).exp()+(b-z).exp()).ln() }

fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|x| !x.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let omega = sigmoid(q[0]);
    let mut mp = Vec::with_capacity(d.t);
    for j in 0..d.t { mp.push(sigmoid(q[1+j])); }
    let gamma_idx=1+d.t; let sigma_idx=gamma_idx+1; let eps_idx=sigma_idx+1;
    let gamma=q[gamma_idx]; let sigma=3.0*sigmoid(q[sigma_idx]);
    let mut dc_omega=0.0; let mut dc_mp=vec![0.0; d.t]; let mut dc_gamma= -gamma/100.0; let mut dc_sigma=0.0;
    let mut dc_eps=vec![0.0; d.m];
    let mut lp = -0.5*(gamma/10.0).powi(2);
    for i in 0..d.m { lp += -0.5*q[eps_idx+i]*q[eps_idx+i]; dc_eps[i] = -q[eps_idx+i]; }
    for i in 0..d.m {
        let eps=sigma*q[eps_idx+i];
        let mut ll=0.0; let mut de=0.0; let mut da=vec![0.0; d.t]; let mut dg=0.0;
        for j in 0..d.t {
            let prev = if j == 0 { 0.0 } else { d.y[i*d.t+j-1] as f64 };
            let eta = (mp[j]/(1.0-mp[j])).ln() + eps + gamma*prev;
            let yy=d.y[i*d.t+j] as f64; let r=yy-sigmoid(eta);
            ll += yy*eta-softplus(eta); de += r; da[j] += r/(mp[j]*(1.0-mp[j])); dg += r*prev;
        }
        let wt: f64;
        if d.observed[i] { lp += omega.ln()+ll; dc_omega += 1.0/omega; wt=1.0; }
        else { let a=omega.ln()+ll; let b=(1.0-omega).ln(); let z=logsumexp(a,b); lp+=z; wt=(a-z).exp(); dc_omega += wt/omega-(1.0-wt)/(1.0-omega); }
        for j in 0..d.t { dc_mp[j] += wt*da[j]; }
        dc_gamma += wt*dg; dc_sigma += wt*de*q[eps_idx+i]; dc_eps[i] += wt*de*sigma;
    }
    // Stan constrained transforms and log-Jacobians, in declaration order.
    lp += log_sigmoid(q[0])+log_sigmoid(-q[0]);
    g[0] = dc_omega*omega*(1.0-omega) + 1.0-2.0*omega;
    for j in 0..d.t { let p=mp[j]; lp += log_sigmoid(q[1+j])+log_sigmoid(-q[1+j]); g[1+j]=dc_mp[j]*p*(1.0-p)+1.0-2.0*p; }
    g[gamma_idx]=dc_gamma;
    let st=sigma/3.0; lp += 3.0_f64.ln()+log_sigmoid(q[sigma_idx])+log_sigmoid(-q[sigma_idx]);
    g[sigma_idx]=dc_sigma*3.0*st*(1.0-st)+1.0-2.0*st;
    for i in 0..d.m { g[eps_idx+i]=dc_eps[i]; }
    Ok(lp)
}

unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output");}unsafe{*out=std::ptr::null_mut();}let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=data.t+data.m+3||got!=want{return Err("dimension/layout mismatch".into());}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast();};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output");}unsafe{*out=std::ptr::null_mut();}let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w;};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let v=eval_model(&b.data,q,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1);}unsafe{*lp=v;};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
