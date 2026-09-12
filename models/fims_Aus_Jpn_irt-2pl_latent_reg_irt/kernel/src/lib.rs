use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { i: usize, j: usize, n: usize, k: usize, ii: Vec<usize>, jj: Vec<usize>, y: Vec<f64>, w_adj: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn usize_field(o: &serde_json::Map<String, Value>, key: &str) -> Result<usize, String> {
    let x = o.get(key).and_then(Value::as_u64).ok_or_else(|| format!("{key} must be a positive integer"))?;
    usize::try_from(x).map_err(|_| format!("{key} too large"))
}
fn int_vec(o: &serde_json::Map<String, Value>, key: &str, n: usize) -> Result<Vec<usize>, String> {
    let a=o.get(key).and_then(Value::as_array).ok_or_else(||format!("{key} must be an array"))?;
    if a.len()!=n { return Err(format!("{key} length mismatch")); }
    a.iter().map(|x| x.as_u64().ok_or_else(||format!("{key} must contain integers")).and_then(|z| usize::try_from(z).map_err(|_|format!("{key} too large")))).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o=v.as_object().ok_or("data must be a JSON object")?;
    let (i,j,n,k)=(usize_field(o,"I")?,usize_field(o,"J")?,usize_field(o,"N")?,usize_field(o,"K")?);
    if i<1 || j<1 || n<1 || k<1 {return Err("dimensions must be positive".into());}
    let mut ii=int_vec(o,"ii",n)?; let mut jj=int_vec(o,"jj",n)?; let yy=int_vec(o,"y",n)?;
    if ii.iter().any(|&x|x<1||x>i)||jj.iter().any(|&x|x<1||x>j)||yy.iter().any(|&x|x>1) {return Err("index or response out of bounds".into());}
    for x in &mut ii {*x-=1}; for x in &mut jj {*x-=1};
    let wa=o.get("W").and_then(Value::as_array).ok_or("W must be a matrix")?;
    if wa.len()!=j {return Err("W row count mismatch".into());}
    let mut w=Vec::with_capacity(j*k);
    for row in wa {let r=row.as_array().ok_or("W must be a matrix")?; if r.len()!=k{return Err("W column count mismatch".into());} for x in r {w.push(x.as_f64().filter(|z|z.is_finite()).ok_or("W must be finite numeric")?);}}
    // This reproduces Stan transformed data; it is not a likelihood summary.
    let mut adj=vec![0.0;2*k]; adj[k]=1.0;
    for col in 1..k {
        let mut min=w[col]; let mut max=w[col]; let mut sum=0.0;
        for row in 0..j {let x=w[row*k+col]; min=min.min(x);max=max.max(x);sum+=x;}
        let mean=sum/(j as f64);
        // Preserve Stan's parsed `minmax_count + W[j,k] == min_w || ...` expression.
        // It assigns the boolean result on each iteration, rather than adding booleans.
        let mut count=0usize; for row in 0..j {let x=w[row*k+col]; count=(((count as f64)+x==min)||x==max) as usize;}
        let scale=if count==j {max-min} else {let ss: f64=(0..j).map(|row|{let z=w[row*k+col]-mean;z*z}).sum(); (ss/((j-1) as f64)).sqrt()*2.0};
        if !scale.is_finite()||scale==0.0{return Err("W adjustment scale is invalid".into());}
        adj[col]=mean; adj[k+col]=scale;
    }
    let mut w_adj=Vec::with_capacity(j*k); for row in 0..j {for col in 0..k {w_adj.push((w[row*k+col]-adj[col])/adj[k+col]);}}
    Ok(Data{i,j,n,k,ii,jj,y:yy.into_iter().map(|x|x as f64).collect(),w_adj})
}
fn expected_layout(d:&Data)->String { let mut x=Vec::new(); for n in 1..=d.i{x.push(format!("alpha.{n}"));} for n in 1..d.i{x.push(format!("beta_free.{n}"));} for n in 1..=d.j{x.push(format!("theta.{n}"));} for n in 1..=d.k{x.push(format!("lambda_adj.{n}"));} x.join("\n") }
fn softplus(x:f64)->f64 {x.max(0.0)+(-x.abs()).exp().ln_1p()}
fn sigmoid(x:f64)->f64 {if x>=0.0 {1.0/(1.0+(-x).exp())} else {let e=x.exp();e/(1.0+e)}}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
    if q.iter().any(|x|!x.is_finite()){return Err("non-finite position".into());} g.fill(0.0);
    let b0=d.i; let t0=2*d.i-1; let l0=t0+d.j; let mut lp=0.0;
    // Stan Math lognormal_lpdf<propto__> retains NEG_LOG_SQRT_TWO_PI once
    // its parameter-dependent summand is included; the lower-bound exp Jacobian
    // cancels the -log(alpha) term.
    const NEG_LOG_SQRT_TWO_PI: f64 = -0.91893853320467274178032973640562;
    for a in 0..d.i {let z=q[a]-1.0; lp += NEG_LOG_SQRT_TWO_PI - 0.5*z*z-q[a]+q[a]; g[a] += -z;}
    // The source explicitly calls normal_lpdf(beta | 0, 3), so stanc emits
    // normal_lpdf<false>: retain its normalizing term for all I beta entries.
    const LOG_3_SQRT_2PI: f64 = 2.017550821872782127045839772;
    let mut last_beta=0.0; for b in 0..d.i-1 {last_beta-=q[b0+b]; lp += -0.5*q[b0+b]*q[b0+b]/9.0 - LOG_3_SQRT_2PI; g[b0+b] += -q[b0+b]/9.0;}
    lp += -0.5*last_beta*last_beta/9.0 - LOG_3_SQRT_2PI; for b in 0..d.i-1 {g[b0+b] += last_beta/9.0;}
    for z in 0..d.k {let x=q[l0+z]; lp += -2.0*(1.0+x*x/3.0).ln(); g[l0+z] += -4.0*x/(3.0+x*x);}
    for person in 0..d.j {let mut mean=0.0;for col in 0..d.k{mean+=d.w_adj[person*d.k+col]*q[l0+col];}let r=q[t0+person]-mean;lp+=-0.5*r*r;g[t0+person]-=r;for col in 0..d.k{g[l0+col]+=r*d.w_adj[person*d.k+col];}}
    for obs in 0..d.n {let item=d.ii[obs];let person=d.jj[obs];let alpha=q[item].exp();let beta=if item+1<d.i {q[b0+item]} else {last_beta};let eta=alpha*q[t0+person]-beta;lp+=d.y[obs]*eta-softplus(eta);let r=d.y[obs]-sigmoid(eta);g[item]+=r*q[t0+person]*alpha;g[t0+person]+=r*alpha;if item+1<d.i{g[b0+item]-=r}else{for b in 0..d.i-1{g[b0+b]+=r;}}}
    Ok(lp)
}
unsafe fn bytes<'a>(p:*const c_char,n:usize)->Result<&'a[u8],String>{if n==0{Ok(&[])}else if p.is_null(){Err("null input".into())}else{Ok(unsafe{slice::from_raw_parts(p.cast(),n)})}}
unsafe fn put_error(p:*mut c_char,cap:usize,s:&str){if !p.is_null()&&cap!=0{let n=s.len().min(cap-1);unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),p.cast(),n);*p.add(n)=0;}}}
fn fatal(p:*mut c_char,cap:usize,s:impl AsRef<str>)->i32{unsafe{put_error(p,cap,s.as_ref())};2}
#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version()->u32{1}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(json:*const c_char,json_len:usize,ndim:usize,layout:*const c_char,layout_len:usize,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null bound output")}unsafe{*out=std::ptr::null_mut()};let answer=catch_unwind(AssertUnwindSafe(||->Result<Bound,String>{let v:Value=serde_json::from_slice(unsafe{bytes(json,json_len)?}).map_err(|e|format!("invalid JSON: {e}"))?;let data=parse_data(v)?;let want=expected_layout(&data);let got=std::str::from_utf8(unsafe{bytes(layout,layout_len)?}).map_err(|_|"layout is not UTF-8")?;if ndim!=2*data.i-1+data.j+data.k||got!=want{return Err("dimension/layout mismatch".into())}Ok(Bound{data,ndim})}));match answer{Ok(Ok(b))=>{unsafe{*out=Box::into_raw(Box::new(b)).cast()};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in bind")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !bound.is_null(){unsafe{drop(Box::from_raw(bound.cast::<Bound>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(bound:*mut c_void,out:*mut *mut c_void,err:*mut c_char,cap:usize)->i32{if out.is_null(){return fatal(err,cap,"null workspace output")}unsafe{*out=std::ptr::null_mut()};let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into())}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in workspace")}}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_:*mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()))}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32{let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into())}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into())}let qq=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let v=eval_model(&b.data,qq,g)?;if !v.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=v};Ok(0)}));match a{Ok(Ok(x))=>x,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"producer panic in evaluate")}}
