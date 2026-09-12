use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data { n: usize, ns: usize, nr: usize, nyo: usize, ny: usize,
 state_region: Vec<usize>, state: Vec<usize>, region: Vec<usize>, year: Vec<usize>, y: Vec<f64> }
struct Bound { data: Data, ndim: usize }
struct Workspace;
fn int(v:&Value,k:&str)->Result<usize,String>{ let x=v.get(k).and_then(Value::as_u64).ok_or_else(||format!("{k} must be a positive integer"))? as usize; if x==0 {Err(format!("{k} must be positive"))} else {Ok(x)} }
fn arr(v:&Value,k:&str,n:usize, max:usize)->Result<Vec<usize>,String>{let a=v.get(k).and_then(Value::as_array).ok_or_else(||format!("{k} must be an array"))?; if a.len()!=n{return Err(format!("{k} length"))}; a.iter().map(|x| {let z=x.as_u64().ok_or_else(||format!("{k} integer"))? as usize;if z==0||z>max{Err(format!("{k} bounds"))}else{Ok(z-1)}}).collect()}
fn parse_data(v:Value)->Result<Data,String>{let n=int(&v,"N")?;let ns=int(&v,"N_states")?;let nr=int(&v,"N_regions")?;let nyo=int(&v,"N_years_obs")?;let ny=int(&v,"N_years")?;let state_region=arr(&v,"state_region_ind",ns,nr)?;let state=arr(&v,"state_ind",n,50)?; if ns!=50{return Err("N_states must be 50 per Stan data constraint".into())};let region=arr(&v,"region_ind",n,10)?;if nr!=10{return Err("N_regions must be 10 per Stan data constraint".into())};let year=arr(&v,"year_ind",n,nyo)?;let ya=v.get("y").and_then(Value::as_array).ok_or("y must be an array")?;if ya.len()!=n{return Err("y length".into())};let mut y=Vec::with_capacity(n);for x in ya{let z=x.as_f64().filter(|z|z.is_finite()&&*z>=0.0&&*z<=1.0).ok_or("y must be finite in [0,1]")?;y.push(z)}Ok(Data{n,ns,nr,nyo,ny,state_region,state,region,year,y})}
fn expected_layout(d:&Data)->String {let mut x=Vec::new();for j in 1..=d.nr{for i in 1..=d.ny{x.push(format!("GP_region_std.{i}.{j}"))}}for j in 1..=d.ns{for i in 1..=d.ny{x.push(format!("GP_state_std.{i}.{j}"))}}for i in 1..=d.nyo{x.push(format!("year_std.{i}"))}for i in 1..=d.ns{x.push(format!("state_std.{i}"))}for i in 1..=d.nr{x.push(format!("region_std.{i}"))}x.push("tot_var".into());for i in 1..17{x.push(format!("prop_var.{i}"))}x.push("mu".into());for s in ["length_GP_region_long","length_GP_state_long","length_GP_region_short","length_GP_state_short"]{x.push(s.into())}x.join("\n")}
#[derive(Clone)]
enum Op { Leaf, Unary(usize, f64), Binary(usize, usize, f64, f64) }
struct Tape { v: Vec<f64>, a: Vec<f64>, op: Vec<Op> }
impl Tape {
    fn new() -> Self { Self { v: vec![], a: vec![], op: vec![] } }
    fn leaf(&mut self, x:f64)->usize { let i=self.v.len(); self.v.push(x); self.a.push(0.0); self.op.push(Op::Leaf); i }
    fn unary(&mut self,x:f64,p:usize,d:f64)->usize { let i=self.v.len(); self.v.push(x);self.a.push(0.0);self.op.push(Op::Unary(p,d));i }
    fn binary(&mut self,x:f64,p:usize,q:usize,dp:f64,dq:f64)->usize { let i=self.v.len();self.v.push(x);self.a.push(0.0);self.op.push(Op::Binary(p,q,dp,dq));i }
    fn add(&mut self,p:usize,q:usize)->usize { self.binary(self.v[p]+self.v[q],p,q,1.0,1.0) }
    fn sub(&mut self,p:usize,q:usize)->usize { self.binary(self.v[p]-self.v[q],p,q,1.0,-1.0) }
    fn mul(&mut self,p:usize,q:usize)->usize { self.binary(self.v[p]*self.v[q],p,q,self.v[q],self.v[p]) }
    fn div(&mut self,p:usize,q:usize)->usize { self.binary(self.v[p]/self.v[q],p,q,1.0/self.v[q],-self.v[p]/(self.v[q]*self.v[q])) }
    fn addc(&mut self,p:usize,c:f64)->usize { self.unary(self.v[p]+c,p,1.0) }
    fn mulc(&mut self,p:usize,c:f64)->usize { self.unary(self.v[p]*c,p,c) }
    fn exp(&mut self,p:usize)->usize { let x=self.v[p].exp();self.unary(x,p,x) }
    fn log(&mut self,p:usize)->usize { self.unary(self.v[p].ln(),p,1.0/self.v[p]) }
    fn sqrt(&mut self,p:usize)->usize { let x=self.v[p].sqrt();self.unary(x,p,0.5/x) }
    fn powi(&mut self,p:usize,n:i32)->usize { let x=self.v[p].powi(n);self.unary(x,p,(n as f64)*self.v[p].powi(n-1)) }
    fn back(&mut self,z:usize) { self.a[z]=1.0; for i in (0..self.v.len()).rev() { let a=self.a[i]; match self.op[i].clone() { Op::Leaf=>{}, Op::Unary(p,d)=>self.a[p]+=a*d, Op::Binary(p,q,dp,dq)=> {self.a[p]+=a*dp;self.a[q]+=a*dq} } } }
}
fn add_normal(t:&mut Tape, lp:usize, x:usize, mu:f64, sigma:usize)->usize {
    let centered=t.addc(x,-mu); let z=t.div(centered,sigma); let sq=t.mul(z,z); let term=t.mulc(sq,-0.5); let out=t.add(lp,term); let ls=t.log(sigma); t.sub(out,ls)
}
fn add_normal_node(t:&mut Tape, lp:usize, x:f64, mu:usize, sigma:usize)->usize {
    let xnode=t.leaf(x); let centered=t.sub(xnode,mu); let z=t.div(centered,sigma); let sq=t.mul(z,z); let term=t.mulc(sq,-0.5); let out=t.add(lp,term); let ls=t.log(sigma); t.sub(out,ls)
}
fn chol(t:&mut Tape,k:&[Vec<usize>])->Vec<Vec<usize>> {
    let n=k.len(); let mut l=vec![vec![0;n];n];
    for i in 0..n { for j in 0..=i { let mut x=k[i][j]; for h in 0..j { let prod=t.mul(l[i][h],l[j][h]); x=t.sub(x,prod); } l[i][j]=if i==j {t.sqrt(x)} else {t.div(x,l[j][j])}; } }
    l
}
fn eval_model(d:&Data,q:&[f64],g:&mut[f64])->Result<f64,String>{
    if q.len()!=933 || q.iter().any(|x|!x.is_finite()) {return Err("nonfinite or wrong position".into());}
    let mut t=Tape::new(); let p:Vec<usize>=q.iter().map(|&x|t.leaf(x)).collect(); let mut at=0;
    let rstd=p[at..at+d.ny*d.nr].to_vec(); at+=d.ny*d.nr;
    let sstd=p[at..at+d.ny*d.ns].to_vec(); at+=d.ny*d.ns;
    let year_std=p[at..at+d.nyo].to_vec(); at+=d.nyo;
    let state_std=p[at..at+d.ns].to_vec(); at+=d.ns;
    let region_std=p[at..at+d.nr].to_vec(); at+=d.nr;
    let tot_q=p[at]; let tot=t.exp(tot_q); at+=1;
    // Pinned Stan Math simplex_constrain uses the ILR/Helmert map then softmax, not stick breaking.
    let mut z: Vec<usize>=(0..17).map(|_|t.leaf(0.0)).collect(); let mut sum_w=t.leaf(0.0);
    for i in (1..17).rev() { let w=t.mulc(p[at+i-1],1.0/((i*(i+1)) as f64).sqrt()); sum_w=t.add(sum_w,w); z[i-1]=t.add(z[i-1],sum_w); let wi=t.mulc(w,i as f64); z[i]=t.sub(z[i],wi); }
    // Stable softmax has the same derivatives; the selected max is intentionally a numeric shift.
    let max_z=z.iter().map(|&u|t.v[u]).fold(f64::NEG_INFINITY,f64::max); let mut ez=Vec::with_capacity(17); let mut denom=t.leaf(0.0); for &u in &z { let shifted=t.addc(u,-max_z); let e=t.exp(shifted); denom=t.add(denom,e); ez.push(e); } let mut prop=Vec::with_capacity(17); let mut log_jac=t.leaf(0.5*(17.0f64).ln()); for &e in &ez { let pi=t.div(e,denom); let lpi=t.log(pi); log_jac=t.add(log_jac,lpi); prop.push(pi); } at+=16;
    let mu=p[at];at+=1;
    let lens=[t.exp(p[at]),t.exp(p[at+1]),t.exp(p[at+2]),t.exp(p[at+3])]; let lenq=[p[at],p[at+1],p[at+2],p[at+3]];
    let mut vars=Vec::with_capacity(17); for &pp in &prop { let z=t.mul(tot,pp); vars.push(t.mulc(z,17.0)); }
    let one=t.leaf(1.0); let half=t.leaf(0.5); let mut lp=t.leaf(0.0);
    for &x in rstd.iter().chain(sstd.iter()).chain(year_std.iter()).chain(state_std.iter()).chain(region_std.iter()) { lp=add_normal(&mut t,lp,x,0.0,one); }
    { let centered=t.addc(mu,-0.5); let z=t.div(centered,half); let sq=t.mul(z,z); let term=t.mulc(sq,-0.5); lp=t.add(lp,term); } // Normal(0.5,0.5): propto drops data-only -log(0.5).
    let ltot=t.log(tot); let term=t.mulc(ltot,2.0); lp=t.add(lp,term); let rate=t.mulc(tot,3.0); lp=t.sub(lp,rate);
    for &x in &prop { let lx=t.log(x); lp=t.add(lp,lx); } lp=t.add(lp,tot_q); lp=t.add(lp,log_jac);
    for (&l,&scale) in lens.iter().zip([8.0,8.0,3.0,3.0].iter()) { let snode=t.leaf(scale); let ratio=t.div(l,snode); let log_length=t.log(l); let shape=t.mulc(log_length,29.0); lp=t.add(lp,shape); let power=t.powi(ratio,30); lp=t.sub(lp,power); }
    for &x in &lenq { lp=t.add(lp,x); }
    let mut kr=vec![vec![0;d.ny];d.ny]; let mut ks=vec![vec![0;d.ny];d.ny];
    for i in 0..d.ny { for j in 0..d.ny { let distance=((i as f64)-(j as f64)).powi(2); let dist=t.leaf(distance);
        let lrl2=t.mul(lens[0],lens[0]); let lrs2=t.mul(lens[2],lens[2]); let lsl2=t.mul(lens[1],lens[1]); let lss2=t.mul(lens[3],lens[3]);
        let za=t.div(dist,lrl2); let za=t.mulc(za,-0.5); let a=t.exp(za); let zb=t.div(dist,lrs2); let zb=t.mulc(zb,-0.5); let b=t.exp(zb); let zc=t.div(dist,lsl2); let zc=t.mulc(zc,-0.5); let c=t.exp(zc); let ze=t.div(dist,lss2); let ze=t.mulc(ze,-0.5); let e=t.exp(ze);
        let ra=t.mul(vars[12],a); let rb=t.mul(vars[14],b); kr[i][j]=t.add(ra,rb); let sa=t.mul(vars[13],c); let sb=t.mul(vars[15],e); ks[i][j]=t.add(sa,sb); if i==j {kr[i][j]=t.addc(kr[i][j],1e-6);ks[i][j]=t.addc(ks[i][j],1e-6);} } }
    let lr=chol(&mut t,&kr); let ls=chol(&mut t,&ks); let mut gr=vec![vec![0;d.nr];d.ny];let mut gs=vec![vec![0;d.ns];d.ny];
    for j in 0..d.nr {for i in 0..d.ny {let mut z=t.leaf(0.0);for h in 0..=i {let prod=t.mul(lr[i][h],rstd[h+d.ny*j]);z=t.add(z,prod);}gr[i][j]=z;}}
    for j in 0..d.ns {for i in 0..d.ny {let mut z=t.leaf(0.0);for h in 0..=i {let prod=t.mul(ls[i][h],sstd[h+d.ny*j]);z=t.add(z,prod);}gs[i][j]=z;}}
    let sy=t.sqrt(vars[0]);let sr=t.sqrt(vars[1]);let se=t.sqrt(vars[16]);let mut ss=Vec::new();for k in 0..10{ss.push(t.sqrt(vars[2+k]));}
    for n in 0..d.n {let a=t.mul(sy,year_std[d.year[n]]);let mut m=t.add(mu,a);let b=t.mul(sr,region_std[d.region[n]]);m=t.add(m,b);let c=t.mul(ss[d.state_region[d.state[n]]],state_std[d.state[n]]);m=t.add(m,c);m=t.add(m,gr[d.year[n]][d.region[n]]);m=t.add(m,gs[d.year[n]][d.state[n]]);lp=add_normal_node(&mut t,lp,d.y[n],m,se);}
    let value=t.v[lp]; if !value.is_finite(){return Err("nonfinite log density".into());} t.back(lp); for (out,&a) in g.iter_mut().zip(t.a.iter().take(q.len())) {*out=a;} Ok(value)
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

#[no_mangle] pub extern "C" fn nutpier_kernel_abi_version() -> u32 { 1 }

#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_bind(
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
        if ndim != want.lines().count() || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    }));
    match answer {
        Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }
        Ok(Err(s)) => fatal(err, cap, s),
        Err(_) => fatal(err, cap, "producer panic in bind"),
    }
}

#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_destroy(bound: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !bound.is_null() {
        unsafe { drop(Box::from_raw(bound.cast::<Bound>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace(
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
        Ok(Err(s)) => fatal(err, cap, s), Err(_) => fatal(err, cap, "producer panic in workspace"),
    }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_workspace_destroy(_: *mut c_void, w: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| if !w.is_null() {
        unsafe { drop(Box::from_raw(w.cast::<Workspace>())); }
    }));
}
#[no_mangle] pub unsafe extern "C" fn nutpier_kernel_evaluate(
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
        Err(_) => fatal(err, cap, "producer panic in evaluate"),
    }
}