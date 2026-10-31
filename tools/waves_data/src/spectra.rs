//! Equations only: Lygre and Krogstad (1986), JPO 16, 2052-2060,
//! doi:10.1175/1520-0485(1986)016<2052:MEEOTD>2.0.CO;2.
//! Station ASCII grammar: NOAA EMC products documentation. Archive bearings
//! were independently checked against GRIB; see data/SCIENCE.md.
use crate::{Result,text,observations};
use chrono::NaiveDateTime;
use serde_json::{json,Value};
use std::{path::Path,fs};
#[derive(Clone)]
pub struct Spectrum{pub epoch:i64,pub id:String,pub lat:f64,pub lon:f64,pub frequency:Vec<f64>,pub from:Vec<f64>,pub efth:Vec<f64>}
pub fn station(path:&Path,begin:i64,end:i64)->Result<Vec<Spectrum>>{
    let text=observations::contents(path)?;let mut lines=text.lines();let header=lines.next().ok_or("empty station spectrum")?;
    let mut quoted=header.split('\'');if quoted.nth(1)!=Some("WAVEWATCH III SPECTRA"){return Err("unrecognized station data header".into());}
    let dims:Vec<usize>=quoted.next().ok_or("missing spectral dimensions")?.split_whitespace().map(str::parse).collect::<std::result::Result<_,_>>()?;
    if dims.len()!=3||dims[0]<2||dims[1]<4||dims[2]!=1{return Err("expected one station with frequency and directional axes".into());}
    let(nf,nd)=(dims[0],dims[1]);
    let read_numbers=|lines:&mut std::str::Lines<'_>,n:usize|->Result<Vec<f64>>{let mut v=Vec::with_capacity(n);while v.len()<n{let line=lines.next().ok_or("truncated station spectrum")?;for word in line.split_whitespace(){let token=word.replace('D',"E");let value=token.parse::<f64>().or_else(|original|{if !token.contains('E')&&!token.contains('e'){if let Some(i)=token.char_indices().skip(1).find_map(|(i,c)|if c=='+'||c=='-'{Some(i)}else{None}){return format!("{}E{}",&token[..i],&token[i..]).parse();}}Err(original)}).map_err(|e|format!("{} spectral number {word:?}: {e}",path.display()))?;v.push(value);}}if v.len()!=n{return Err("station spectral row overrun".into());}Ok(v)};
    let frequency=read_numbers(&mut lines,nf)?;let theta=read_numbers(&mut lines,nd)?;
    if frequency.windows(2).any(|w|w[1]<=w[0])||frequency[0]<=0.{return Err("station frequencies must increase".into());}
    // Operational station coordinates are nautical TOWARD radians. Validate
    // the orientation against separately encoded GRIB directions, not the
    // older EMC webpage's Cartesian description.
    let from:Vec<f64>=theta.iter().map(|t|(180.+t.to_degrees()).rem_euclid(360.)).collect();let mut spectra=Vec::new();
    while let Some(line)=lines.next(){if line.trim().is_empty(){continue}let when=NaiveDateTime::parse_from_str(line.trim(),"%Y%m%d %H%M%S")?.and_utc().timestamp();
        let station=lines.next().ok_or("missing station record")?;let mut parts=station.split('\'');let id=parts.nth(1).ok_or("quoted station id missing")?.trim().to_string();
        let environment=parts.next().ok_or("station environment missing")?;
        // NOAA station ASCII uses adjacent fixed-width f7.2 latitude/longitude;
        // a three-digit western longitude can touch the latitude's last digit.
        let env:Vec<f64>=if environment.len()>=14&&environment.is_ascii(){vec![environment[..7].trim().parse().map_err(|e|format!("{} latitude {:?}: {e}",path.display(),&environment[..7]))?,environment[7..14].trim().parse().map_err(|e|format!("{} longitude {:?}: {e}",path.display(),&environment[7..14]))?]}else{return Err("station coordinate fields truncated".into());};
        let values=read_numbers(&mut lines,nf*nd)?;
        if when<begin||when>end{continue}
        if values.iter().any(|v|!v.is_finite()||*v<0.){return Err("negative or nonfinite station spectral variance".into());}
        // ASCII output has Fortran frequency-first storage: NF words per direction.
        let efth=(0..nf*nd).map(|i|values[(i%nd)*nf+i/nd]).collect();
        spectra.push(Spectrum{epoch:when,id,lat:env[0],lon:env[1],frequency:frequency.clone(),from:from.clone(),efth});
    }Ok(spectra)
}
pub fn widths(f:&[f64])->Vec<f64>{(0..f.len()).map(|i|if i==0{0.5*(f[1]-f[0]*f[0]/f[1])}else if i+1==f.len(){0.5*(f[i]*f[i]/f[i-1]-f[i-1])}else{0.5*(f[i+1]-f[i-1])}).collect()}
pub fn moments(s:&Spectrum)->Value{
    let nd=s.from.len();let df=widths(&s.frequency);let dtheta=2.*std::f64::consts::PI/nd as f64;let mut m=[0.;3];let mut fenergy=vec![0.;s.frequency.len()];let(mut c,mut si)=(0.,0.);
    for(i,row)in s.efth.chunks_exact(nd).enumerate(){for(j,&e)in row.iter().enumerate(){let v=e*df[i]*dtheta;m[0]+=v;m[1]+=v*s.frequency[i];m[2]+=v*s.frequency[i].powi(2);fenergy[i]+=e*dtheta;c+=v*s.from[j].to_radians().cos();si+=v*s.from[j].to_radians().sin();}}
    let peak=fenergy.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).map(|(i,_)|i).unwrap_or(0);
    let(mut pc,mut ps)=(0.,0.);for j in 0..nd{pc+=s.efth[peak*nd+j]*s.from[j].to_radians().cos();ps+=s.efth[peak*nd+j]*s.from[j].to_radians().sin();}
    json!({"hs":4.*m[0].sqrt(),"tp":1./s.frequency[peak],"tm01":if m[1]>0.{m[0]/m[1]}else{0.},"tm02":if m[2]>0.{(m[0]/m[2]).sqrt()}else{0.},"dp":ps.atan2(pc).to_degrees().rem_euclid(360.),"dir":si.atan2(c).to_degrees().rem_euclid(360.),"dspr":if m[0]>0.{(2.*(1.-c.hypot(si)/m[0]).max(0.)).sqrt().to_degrees()}else{0.},"frequency":s.frequency,"energy":fenergy})
}
// Small explicit complex arithmetic avoids a second spectral package.
type C=(f64,f64);
fn mul(a:C,b:C)->C{(a.0*b.0-a.1*b.1,a.0*b.1+a.1*b.0)}
fn conj(a:C)->C{(a.0,-a.1)}
fn phase(r:f64,a:f64)->C{(r*a.cos(),r*a.sin())}
pub fn mem(r1:f64,r2:f64,a1:f64,a2:f64,from:&[f64])->Option<Vec<f64>>{
    if ![r1,r2,a1,a2].iter().all(|v|v.is_finite())||!(0.0..1.0).contains(&r1)||!(0.0..=1.0).contains(&r2){return None}
    let c1=phase(r1,a1.to_radians());let c2=phase(r2,2.*a2.to_radians());let q=mul(c2,conj(c1));let p1=((c1.0-q.0)/(1.-r1*r1),(c1.1-q.1)/(1.-r1*r1));let q=mul(p1,c1);let p2=(c2.0-q.0,c2.1-q.1);
    let variance=1.-mul(p1,conj(c1)).0-mul(p2,conj(c2)).0;if variance<=0.||!variance.is_finite(){return None}
    let mut v:Vec<f64>=from.iter().map(|a|{let q=mul(p1,phase(1.,-a.to_radians()));let r=mul(p2,phase(1.,-2.*a.to_radians()));variance/((1.-q.0-r.0).powi(2)+(q.1+r.1).powi(2))}).collect();
    let mass=v.iter().sum::<f64>()*2.*std::f64::consts::PI/from.len() as f64;if mass<=0.||!mass.is_finite(){return None}for e in &mut v{*e/=mass;}Some(v)
}
pub fn inspect(r:&Value,out:&Path)->Result<Value>{let s=station(Path::new(text(r,"input")?),i64::MIN,i64::MAX)?;let rows:Vec<_>=s.iter().map(|s|json!({"time":s.epoch,"station":s.id,"lat":s.lat,"lon":s.lon,"moments":moments(s)})).collect();fs::write(out.join("spectra-summary.json"),serde_json::to_vec_pretty(&rows)?)?;Ok(json!({"status":"complete","records":rows.len()}))}
