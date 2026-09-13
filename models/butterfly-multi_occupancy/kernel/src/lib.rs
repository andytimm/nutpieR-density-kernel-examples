use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { j: usize, k: f64, n: usize, s: usize, x: Vec<i32> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn int_field(o: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    o.get(key).and_then(Value::as_u64).and_then(|x| usize::try_from(x).ok())
        .filter(|&x| x >= 1).ok_or_else(|| format!("{key} must be positive integer"))
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let j = int_field(o, "J")?; let k_u = int_field(o, "K")?;
    let n = int_field(o, "n")?; let s = int_field(o, "S")?;
    if s < n { return Err("S must be >= n".into()); }
    let rows = o.get("X").and_then(Value::as_array).ok_or("X must be an array")?;
    if rows.len() != n { return Err("X row count mismatch".into()); }
    let mut x = Vec::with_capacity(n.checked_mul(j).ok_or("X size overflow")?);
    for row in rows {
        let a = row.as_array().ok_or("X rows must be arrays")?;
        if a.len() != j { return Err("X column count mismatch".into()); }
        for z in a {
            let z = z.as_u64().and_then(|q| i32::try_from(q).ok()).ok_or("X must contain integers")?;
            if z < 0 || z as usize > k_u { return Err("X must be in 0..K".into()); }
            x.push(z);
        }
    }
    Ok(Data { j, k: k_u as f64, n, s, x })
}
fn expected_layout(d: &Data) -> String {
    let mut names = vec!["alpha".to_string(), "beta".to_string(), "Omega".to_string(), "rho_uv".to_string()];
    for i in 1..=2 { names.push(format!("sigma_uv.{i}")); }
    for i in 1..=d.s { names.push(format!("uv1.{i}")); }
    for i in 1..=d.s { names.push(format!("uv2.{i}")); }
    names.join("\n")
}
#[inline] fn softplus(x: f64) -> f64 { x.max(0.0) + (-x.abs()).exp().ln_1p() }
#[inline] fn sigmoid(x: f64) -> f64 { if x >= 0.0 { 1.0 / (1.0 + (-x).exp()) } else { let e=x.exp(); e/(1.0+e) } }
#[inline] fn log_sigmoid(x: f64) -> f64 { -softplus(-x) }
#[inline] fn logsumexp(a: f64, b: f64) -> f64 { let m=a.max(b); m + ((a-m).exp()+(b-m).exp()).ln() }
#[inline] fn log_choose(n: usize, k: usize) -> f64 { let k=k.min(n-k); (1..=k).map(|i| ((n-k+i) as f64).ln()-(i as f64).ln()).sum() }

fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    if q.iter().any(|z| !z.is_finite()) { return Err("non-finite position".into()); }
    g.fill(0.0);
    let alpha=q[0]; let beta=q[1];
    let omega=sigmoid(q[2]); let rho=2.0*sigmoid(q[3])-1.0;
    let s1=q[4].exp(); let s2=q[5].exp();
    if !omega.is_finite() || !rho.is_finite() || !s1.is_finite() || !s2.is_finite() || omega <= 0.0 || omega >= 1.0 || s1 <= 0.0 || s2 <= 0.0 { return Err("transform domain".into()); }
    let det=1.0-rho*rho;
    if det <= 0.0 || !det.is_finite() { return Err("correlation domain".into()); }
    let mut lp=0.0;
    // cauchy(0, 2.5), with propto=true: scale-only normalizers omitted.
    let scale2=6.25;
    for (idx, v) in [(0,alpha),(1,beta)] { lp -= (1.0+v*v/scale2).ln(); g[idx] -= 2.0*v/(scale2+v*v); }
    // beta(2,2) priors on constrained Omega and (rho+1)/2.
    lp += omega.ln() + (-omega).ln_1p();
    let domega_prior = 1.0/omega - 1.0/(1.0-omega);
    let rhalf=(rho+1.0)*0.5;
    lp += rhalf.ln() + (-rhalf).ln_1p();
    let drho_prior = 1.0/(rho+1.0) - 1.0/(1.0-rho);
    // cauchy priors for constrained scales and bivariate normal random effects.
    for idx in [4usize,5usize] { let v=q[idx].exp(); lp -= (1.0+v*v/scale2).ln(); }
    let mut ds1=-2.0*s1/(scale2+s1*s1) - (d.s as f64)/s1; let mut ds2=-2.0*s2/(scale2+s2*s2) - (d.s as f64)/s2; let mut drho=drho_prior;
    lp += -(d.s as f64)*(s1.ln()+s2.ln()) - 0.5*(d.s as f64)*det.ln() - (d.s as f64)*(2.0 * std::f64::consts::PI).ln();
    drho += (d.s as f64)*rho/det;
    for i in 0..d.s {
        let u1=q[6+i]; let u2=q[6+d.s+i]; let x=u1/s1; let y=u2/s2;
        let numer=x*x-2.0*rho*x*y+y*y;
        lp -= 0.5*numer/det;
        let du1=-(x-rho*y)/(det*s1); let du2=-(y-rho*x)/(det*s2);
        g[6+i] += du1; g[6+d.s+i] += du2;
        ds1 += (x*x-rho*x*y)/(det*s1);
        ds2 += (y*y-rho*x*y)/(det*s2);
        drho += x*y/det-rho*numer/(det*det);
    }
    // Likelihood, fused with its derivatives. `X` is only copied/repacked at bind.
    for i in 0..d.s {
        let p=alpha+q[6+i]; let t=beta+q[6+d.s+i]; let sp=sigmoid(p); let st=sigmoid(t);
        let a=log_sigmoid(p)-d.k*softplus(t); // zero-detection available term
        let b=log_sigmoid(-p);                // zero-detection unavailable term
        let r=sigmoid(a-b);                   // conditional availability for zero observation
        let (dp,dt,do_) = if i < d.n {
            let mut dp=0.0; let mut dt=0.0;
            for jj in 0..d.j {
                let obs=d.x[i*d.j+jj] as f64;
                if obs > 0.0 { lp += log_choose(d.k as usize, obs as usize) + log_sigmoid(p)+obs*t-d.k*softplus(t); dp += 1.0-sp; dt += obs-d.k*st; }
                else { lp += logsumexp(a,b); dp += r-sp; dt += -r*d.k*st; }
            }
            lp += omega.ln(); (dp,dt,1.0/omega)
        } else {
            let c=(-omega).ln_1p(); let dd=omega.ln()+(d.j as f64)*logsumexp(a,b);
            let w=sigmoid(dd-c); lp += logsumexp(c,dd);
            (w*(d.j as f64)*(r-sp), w*(d.j as f64)*(-r*d.k*st), w/omega-(1.0-w)/(1.0-omega))
        };
        g[0]+=dp; g[1]+=dt; g[6+i]+=dp; g[6+d.s+i]+=dt;
        // Accumulate constrained Omega derivative before its transform below.
        g[2]+=do_;
    }
    // Chain rules plus Jacobians in exact constrained-parameter declaration order.
    g[2] = (g[2]+domega_prior)*omega*(1.0-omega) + (1.0-2.0*omega);
    lp += q[2] - 2.0*softplus(q[2]);
    g[3] = drho * 2.0*sigmoid(q[3])*(1.0-sigmoid(q[3])) - rho;
    lp += 2.0_f64.ln() + q[3] - 2.0*softplus(q[3]);
    g[4] = ds1*s1 + 1.0; g[5] = ds2*s2 + 1.0; lp += q[4]+q[5];
    if !lp.is_finite() || g.iter().any(|z| !z.is_finite()) { return Err("non-finite result".into()); }
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> { if n==0 {Ok(&[])} else if p.is_null(){Err("null input".into())} else {Ok(unsafe{slice::from_raw_parts(p.cast(),n)})} }
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32 { if out.is_null(){return fatal(err,cap,"null bound output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let d=parse_data(v)?;let want=expected_layout(&d);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=6+2*d.s||got!=want{return Err("dimension/layout mismatch".into())};Ok(Bound{data:d,ndim})}));match a{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(b:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !b.is_null(){unsafe{drop(Box::from_raw(b.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(b:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")};unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if b.is_null(){return Err("null bound handle".into())};Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(b:*mut c_void,w:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,g:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if b.is_null()||w.is_null()||lp.is_null()||g.is_null(){return Err("null evaluation handle/output".into())};let b=unsafe{&*b.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())};let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(g,ndim)};let x=eval_model(&b.data,q,g)?;if !x.is_finite()||g.iter().any(|z|!z.is_finite()){return Ok(1)};unsafe{*lp=x};Ok(0)}));match a{Ok(Ok(z))=>z,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density kernel panic in evaluate")}}
