use serde_json::Value;
use std::{ffi::{c_char, c_void}, panic::{catch_unwind, AssertUnwindSafe}, slice};

struct Data {
    n: usize,
    j: usize,
    county: Vec<usize>, // zero-based copies of supplied one-based county_idx
    floor: Vec<f64>,
    y: Vec<f64>,
}
struct Bound { data: Data, ndim: usize }
struct Workspace;

fn finite_num(v: &Value, key: &str) -> Result<f64, String> {
    v.get(key).and_then(Value::as_f64).filter(|x| x.is_finite())
        .ok_or_else(|| format!("{key} must be finite numeric"))
}
fn nonnegative_usize(v: &Value, key: &str) -> Result<usize, String> {
    let x = finite_num(v, key)?;
    if x < 0.0 || x.fract() != 0.0 || x > usize::MAX as f64 {
        return Err(format!("{key} must be a nonnegative integer"));
    }
    Ok(x as usize)
}
fn finite_array(v: &Value, key: &str, n: usize) -> Result<Vec<f64>, String> {
    let a = v.get(key).and_then(Value::as_array).ok_or_else(|| format!("{key} must be an array"))?;
    if a.len() != n { return Err(format!("{key} has wrong length")); }
    a.iter().map(Value::as_f64).enumerate().map(|(i, x)|
        x.filter(|z| z.is_finite()).ok_or_else(|| format!("{key}[{}] must be finite numeric", i + 1))
    ).collect()
}
fn parse_data(v: Value) -> Result<Data, String> {
    let o = v.as_object().ok_or("data must be a JSON object")?;
    let root = Value::Object(o.clone());
    let n = nonnegative_usize(&root, "N")?;
    let j = nonnegative_usize(&root, "J")?;
    let county_v = root.get("county_idx").and_then(Value::as_array).ok_or("county_idx must be an array")?;
    if county_v.len() != n { return Err("county_idx has wrong length".into()); }
    let mut county = Vec::with_capacity(n);
    for (i, x) in county_v.iter().enumerate() {
        let x = x.as_f64().filter(|z| z.is_finite() && *z >= 1.0 && z.fract() == 0.0 && *z <= j as f64)
            .ok_or_else(|| format!("county_idx[{}] must be an integer in 1..J", i + 1))?;
        county.push(x as usize - 1);
    }
    let floor = finite_array(&root, "floor_measure", n)?;
    let y = finite_array(&root, "log_radon", n)?;
    Ok(Data { n, j, county, floor, y })
}
fn expected_layout(d: &Data) -> String {
    let mut names = vec!["sigma_y".to_string(), "sigma_alpha".to_string(), "sigma_beta".to_string()];
    names.extend((1..=d.j).map(|i| format!("alpha.{i}")));
    names.extend((1..=d.j).map(|i| format!("beta.{i}")));
    names.push("mu_alpha".into()); names.push("mu_beta".into());
    names.join("\n")
}

// Exact Stan target with propto=true,jacobian=true.  The three positive scales
// use exp unconstraining transforms, so each receives its log-Jacobian.
fn eval_model(d: &Data, q: &[f64], g: &mut [f64]) -> Result<f64, String> {
    let sy = q[0].exp(); let sa = q[1].exp(); let sb = q[2].exp();
    if !sy.is_finite() || !sa.is_finite() || !sb.is_finite() { return Err("non-finite scale transform".into()); }
    let a0 = 3; let b0 = a0 + d.j; let mua_i = b0 + d.j; let mub_i = mua_i + 1;
    g.fill(0.0);
    let mua = q[mua_i]; let mub = q[mub_i];
    let inv_sy2 = 1.0 / (sy * sy); let inv_sa2 = 1.0 / (sa * sa); let inv_sb2 = 1.0 / (sb * sb);
    let mut lp = -0.5 * sy * sy + q[0] - 0.5 * sa * sa + q[1] - 0.5 * sb * sb + q[2]
        - 0.5 * (mua / 10.0).powi(2) - 0.5 * (mub / 10.0).powi(2)
        // This source uses explicit normal_lpdf for each observation; unlike the
        // sampling statements above it retains the normal constant under propto.
        - 0.5 * (2.0 * std::f64::consts::PI).ln() * d.n as f64;
    let mut sum_a2 = 0.0; let mut sum_b2 = 0.0; let mut sum_r2 = 0.0;
    for k in 0..d.j {
        let da = q[a0 + k] - mua; let db = q[b0 + k] - mub;
        sum_a2 += da * da; sum_b2 += db * db;
        lp += -0.5 * da * da * inv_sa2 - q[1] - 0.5 * db * db * inv_sb2 - q[2];
        g[a0 + k] = -da * inv_sa2;
        g[b0 + k] = -db * inv_sb2;
        g[mua_i] += da * inv_sa2;
        g[mub_i] += db * inv_sb2;
    }
    for n in 0..d.n {
        let k = d.county[n]; let r = d.y[n] - (q[a0 + k] + d.floor[n] * q[b0 + k]);
        sum_r2 += r * r;
        lp += -0.5 * r * r * inv_sy2 - q[0];
        g[a0 + k] += r * inv_sy2;
        g[b0 + k] += d.floor[n] * r * inv_sy2;
    }
    g[0] = -sy * sy + 1.0 - d.n as f64 + sum_r2 * inv_sy2;
    g[1] = -sa * sa + 1.0 - d.j as f64 + sum_a2 * inv_sa2;
    g[2] = -sb * sb + 1.0 - d.j as f64 + sum_b2 * inv_sb2;
    g[mua_i] -= mua / 100.0; g[mub_i] -= mub / 100.0;
    Ok(lp)
}

unsafe fn bytes<'a>(p: *const c_char, n: usize) -> Result<&'a [u8], String> {
    if n == 0 { Ok(&[]) } else if p.is_null() { Err("null input".into()) }
    else { Ok(unsafe { slice::from_raw_parts(p.cast(), n) }) }
}
unsafe fn put_error(p: *mut c_char, cap: usize, s: &str) {
    if !p.is_null() && cap != 0 { let n = s.len().min(cap - 1);
        unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), p.cast(), n); *p.add(n) = 0; } }
}
fn fatal(p: *mut c_char, cap: usize, s: impl AsRef<str>) -> i32 { unsafe { put_error(p, cap, s.as_ref()) }; 2 }
#[no_mangle] pub extern "C" fn nutpier_density_evaluator_abi_version() -> u32 { 1 }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_bind(json: *const c_char, json_len: usize, ndim: usize, layout: *const c_char, layout_len: usize, out: *mut *mut c_void, err: *mut c_char, cap: usize) -> i32 {
    if out.is_null() { return fatal(err, cap, "null bound output"); } unsafe { *out = std::ptr::null_mut(); }
    let answer = catch_unwind(AssertUnwindSafe(|| -> Result<Bound, String> {
        let v: Value = serde_json::from_slice(unsafe { bytes(json, json_len)? }).map_err(|e| format!("invalid JSON: {e}"))?;
        let data = parse_data(v)?; let want = expected_layout(&data);
        let got = std::str::from_utf8(unsafe { bytes(layout, layout_len)? }).map_err(|_| "layout is not UTF-8")?;
        if ndim != 2 * data.j + 5 || got != want { return Err("dimension/layout mismatch".into()); }
        Ok(Bound { data, ndim })
    })); match answer { Ok(Ok(b)) => { unsafe { *out = Box::into_raw(Box::new(b)).cast(); }; 0 }, Ok(Err(s)) => fatal(err,cap,s), Err(_) => fatal(err,cap,"density evaluator panic in bind") }
}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_destroy(bound: *mut c_void) { let _=catch_unwind(AssertUnwindSafe(|| if !bound.is_null() { unsafe { drop(Box::from_raw(bound.cast::<Bound>())); } })); }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace(bound: *mut c_void,out: *mut *mut c_void,err: *mut c_char,cap: usize)->i32 { if out.is_null(){return fatal(err,cap,"null workspace output");} unsafe{*out=std::ptr::null_mut();} let a=catch_unwind(AssertUnwindSafe(||->Result<*mut c_void,String>{if bound.is_null(){return Err("null bound handle".into());}Ok(Box::into_raw(Box::new(Workspace)).cast())}));match a{Ok(Ok(w))=>{unsafe{*out=w};0},Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in workspace")} }
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_workspace_destroy(_: *mut c_void,w:*mut c_void){let _=catch_unwind(AssertUnwindSafe(||if !w.is_null(){unsafe{drop(Box::from_raw(w.cast::<Workspace>()));}}));}
#[no_mangle] pub unsafe extern "C" fn nutpier_density_evaluator_evaluate(bound:*mut c_void,workspace:*mut c_void,q:*const f64,ndim:usize,lp:*mut f64,gradient:*mut f64,err:*mut c_char,cap:usize)->i32 {let a=catch_unwind(AssertUnwindSafe(||->Result<i32,String>{if bound.is_null()||workspace.is_null()||lp.is_null()||gradient.is_null(){return Err("null evaluation handle/output".into());}let b=unsafe{&*bound.cast::<Bound>()};if ndim!=b.ndim||(ndim!=0&&q.is_null()){return Err("evaluation dimension".into());}let q=if ndim==0{&[]}else{unsafe{slice::from_raw_parts(q,ndim)}};let g=unsafe{slice::from_raw_parts_mut(gradient,ndim)};let value=eval_model(&b.data,q,g)?;if !value.is_finite()||g.iter().any(|x|!x.is_finite()){return Ok(1)}unsafe{*lp=value};Ok(0)}));match a{Ok(Ok(s))=>s,Ok(Err(s))=>fatal(err,cap,s),Err(_)=>fatal(err,cap,"density evaluator panic in evaluate")}}
