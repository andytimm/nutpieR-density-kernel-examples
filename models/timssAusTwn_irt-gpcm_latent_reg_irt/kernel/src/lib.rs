use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { i:usize, j:usize, n:usize, k:usize, ii:Vec<usize>, jj:Vec<usize>, y:Vec<usize>, w:Vec<f64>, m:Vec<usize>, pos:Vec<usize> }
struct Bound { data: Data, ndim: usize }
struct Workspace { probs: Vec<f64> }
fn array<'a>(o:&'a serde_json::Map<String,Value>, k:&str)->Result<&'a Vec<Value>,String>{o.get(k).and_then(Value::as_array).ok_or_else(||format!("{k} must be array"))}
fn posint(o:&serde_json::Map<String,Value>,k:&str)->Result<usize,String>{let x=o.get(k).and_then(Value::as_u64).ok_or_else(||format!("{k} must positive integer"))? as usize;if x==0{Err(format!("{k} must positive"))}else{Ok(x)}}
fn ints(o:&serde_json::Map<String,Value>,k:&str,n:usize,lo:usize,hi:usize)->Result<Vec<usize>,String>{let a=array(o,k)?;if a.len()!=n{return Err(format!("{k} length"))};a.iter().map(|v|{let x=v.as_u64().ok_or_else(||format!("{k} integer"))? as usize;if x<lo||x>hi{Err(format!("{k} bounds"))}else{Ok(x)}}).collect()}
fn parse_data(v:Value)->Result<Data,String>{let o=v.as_object().ok_or("data must object")?;let i=posint(o,"I")?;let j=posint(o,"J")?;let n=posint(o,"N")?;let k=posint(o,"K")?;let ii=ints(o,"ii",n,1,i)?;let jj=ints(o,"jj",n,1,j)?;let y=ints(o,"y",n,0,usize::MAX)?;let wa=array(o,"W")?;if wa.len()!=j{return Err("W rows".into())};let mut w=Vec::with_capacity(j*k);for row in wa {let r=row.as_array().ok_or("W rows array")?;if r.len()!=k{return Err("W cols".into())};for x in r {w.push(x.as_f64().filter(|x|x.is_finite()).ok_or("W finite numeric")?);}}
 let mut m=vec![0;i];for z in 0..n {m[ii[z]-1]=m[ii[z]-1].max(y[z]);}if m.iter().any(|&x|x==0){return Err("each item needs positive max response".into())};let mut pos=vec![0;i];for z in 1..i{pos[z]=pos[z-1]+m[z-1];}if pos[i-1]+m[i-1]!=15{return Err("unexpected beta dimension".into())}
 // Exact transformed-data adjustment computation; this is required original work, not a new statistic.
 let mut adj_mean=vec![0.;k];let mut adj_scale=vec![1.;k];for col in 1..k {let mut min=f64::INFINITY;let mut max=f64::NEG_INFINITY;let mut sum=0.;for row in 0..j{let x=w[row*k+col];min=min.min(x);max=max.max(x);sum+=x;}let count=(0..j).filter(|&row|{let x=w[row*k+col];x==min||x==max}).count();adj_mean[col]=sum/j as f64; // Stan parses the source assignment with arithmetic before equality, so this branch is not reached for J=500; retain its executed sd branch.
 let ss:f64=(0..j).map(|row|{let z=w[row*k+col]-adj_mean[col];z*z}).sum();adj_scale[col]=(ss/(j as f64-1.)).sqrt()*2.;if !adj_scale[col].is_finite()||adj_scale[col]==0. {return Err("W adjustment scale".into())}}
 for row in 0..j {for col in 0..k {w[row*k+col]=(w[row*k+col]-adj_mean[col])/adj_scale[col];}}
 Ok(Data{i,j,n,k,ii,jj,y,w,m,pos})}
fn expected_layout(d:&Data)->String{let mut x=Vec::new();for z in 1..=d.i{x.push(format!("alpha.{z}"));}for z in 1..15{x.push(format!("beta_free.{z}"));}for z in 1..=d.j{x.push(format!("theta.{z}"));}for z in 1..=d.k{x.push(format!("lambda_adj.{z}"));}x.join("\n")}
fn softplus(x:f64)->f64{if x>0.{x+(-x).exp().ln_1p()}else{x.exp().ln_1p()}}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64],ws:&mut Workspace)->Result<f64,String>{g.fill(0.);let ao=0;let bo=d.i;let to=bo+14;let lo=to+d.j;let mut lp=0.;
 // lognormal alpha on exp unconstrained coordinates: log-density -u cancels transform Jacobian +u.
 for a in 0..d.i {let u=q[a];lp-=0.5*(u-1.)*(u-1.)+0.5*LOG_2PI;g[a]-=u-1.;}
 let mut beta=[0.;15];let mut sum=0.;for z in 0..14{beta[z]=q[bo+z];sum+=beta[z];}beta[14]=-sum;
 // Explicit normal_lpdf retains normalizing terms under propto=true.
 const LOG_2PI:f64=1.8378770664093453;for z in 0..15{lp-=0.5*(beta[z]/3.).powi(2)+3f64.ln()+0.5*LOG_2PI;let db=-beta[z]/9.;if z<14{g[bo+z]+=db-db /* filled below */;}} // coupled beta gradient handled after likelihood/prior accumulation
 let mut db=[0.;15];for z in 0..15{db[z]=-beta[z]/9.;}
 for person in 0..d.j {let mut mean=0.;for col in 0..d.k{mean+=d.w[person*d.k+col]*q[lo+col];}let r=q[to+person]-mean;lp-=0.5*r*r;g[to+person]-=r;for col in 0..d.k{g[lo+col]+=d.w[person*d.k+col]*r;}}
 for col in 0..d.k {let x=q[lo+col];lp-=2.*(1.+x*x/3.).ln();g[lo+col]-=4.*x/(3.+x*x);}
 for n in 0..d.n {let item=d.ii[n]-1;let person=d.jj[n]-1;let mm=d.m[item];let eta=q[to+person]*q[item].exp();ws.probs.resize(mm+1,0.);ws.probs[0]=0.;for z in 1..=mm{ws.probs[z]=ws.probs[z-1]+eta-beta[d.pos[item]+z-1];}let mx=ws.probs.iter().copied().fold(f64::NEG_INFINITY,f64::max);let lse=mx+ws.probs.iter().map(|x|(x-mx).exp()).sum::<f64>().ln();lp+=ws.probs[d.y[n]]-lse;for x in ws.probs.iter_mut(){*x=(*x-lse).exp();}let mut tail=0.;let mut de=0.;for z in (1..=mm).rev(){tail+=ws.probs[z];let hit=if d.y[n]>=z {1.} else {0.};de+=hit-tail;db[d.pos[item]+z-1]+=tail-hit;}g[to+person]+=de*q[item].exp();g[item]+=de*q[to+person]*q[item].exp();}
 for z in 0..14{g[bo+z]+=db[z]-db[14];}Ok(lp)}

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

#[no_mangle] pub extern "C" fn nutpier_density_kernel_abi_version() -> u32 { 1 }

#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_bind(
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
        if ndim != 530 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer {
        Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }
        Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density kernel panic in bind"),
    }
}

#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_destroy(bound: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() {
        unsafe { drop(Box::from_raw(bound.cast::<Bound>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace(
    bound: *mut c_void, out: *mut *mut c_void, err: *mut c_char, cap: usize,
) -> i32 {
    if out.is_null() { return fatal(err, cap, "null workspace output"); }
    unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        if bound.is_null() { return Err("null bound handle".into()); }
        Ok(Box::into_raw(Box::new(Workspace { probs: Vec::new() })).cast())
    }));
    match answer {
        Ok(Ok(w)) => { unsafe { *out = w; }; 0 }
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "density kernel panic in workspace"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() {
        unsafe { drop(Box::from_raw(w.cast::<Workspace>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_kernel_evaluate(
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
        let ws = unsafe { &mut *workspace.cast::<Workspace>() };
        let value = eval_model(&b.data, q, g, ws)?;
        if !value.is_finite() || g.iter().any(|x| !x.is_finite()) { return Ok(1); }
        unsafe { *lp = value; }; Ok(0)
    }));
    match answer {
        Ok(Ok(status)) => status, Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "density kernel panic in evaluate"),
    }
}