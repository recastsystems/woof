//! WRF 4.6.1 module_physics_init.F:5444-5628 native WRF aerosol record reader and grid interpolation.

#[used]
static SOURCE_REV: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
use std::{env,fs::File,io::{self,Read,Write,BufReader,BufWriter},thread};
// Input preparation may cap its CPU budget independently of the model's OMP
// setting. Large decoding and interpolation stages share this bounded pool.
fn workers()->usize {
 env::var("GPUWM_AEROSOL_GRID_THREADS").ok().and_then(|v|v.parse::<usize>().ok())
  .unwrap_or_else(||thread::available_parallelism().map_or(2,usize::from).min(8)).clamp(2,32)
}
fn parallel_fill<T:Send,F:Fn(usize,&mut T)+Sync>(out:&mut [T],f:F) {
 let count=out.len();if count==0{return;}let chunk=count.div_ceil(workers().min(count));
 thread::scope(|scope|{for (part,items) in out.chunks_mut(chunk).enumerate(){
  let fill=&f;scope.spawn(move||{for (local,item) in items.iter_mut().enumerate(){fill(part*chunk+local,item);}});
 }});
}
fn bad(s:&str)->io::Error{io::Error::new(io::ErrorKind::InvalidData,s)}
fn record(r:&mut impl Read,be:bool)->io::Result<Vec<u8>>{
 let mut out=Vec::new();loop{let mut b=[0u8;4];r.read_exact(&mut b)?;let n=if be{i32::from_be_bytes(b)}else{i32::from_le_bytes(b)};let count=n.unsigned_abs() as usize;
 let start=out.len();out.resize(start.checked_add(count).ok_or_else(||bad("record length overflow"))?,0);r.read_exact(&mut out[start..])?;r.read_exact(&mut b)?;
 let end=if be{i32::from_be_bytes(b)}else{i32::from_le_bytes(b)};if end.unsigned_abs() as usize!=count{return Err(bad("Fortran record markers disagree"));}if n>=0{break;}}
 Ok(out)
}
fn floats(v:Vec<u8>,be:bool,count:usize)->io::Result<Vec<f32>>{
 if v.len()!=count*4{return Err(bad("WRF aerosol record extent disagrees with header"));}
 let mut out=vec![0.;count];parallel_fill(&mut out,|i,x|{let word=v[i*4..i*4+4].try_into().unwrap();*x=if be{f32::from_be_bytes(word)}else{f32::from_le_bytes(word)};});Ok(out)
}
fn interp(v:&[f32],want:f32)->(usize,usize,f32,f32){
 let n=v.len();let mut pair=None;for i in 0..n-1{if v[i]<=want && v[i+1]>want{pair=Some((i,i+1,v[i],v[i+1]));break;}}
 let (a,b,x,y)=pair.unwrap_or_else(||if want<v[0]{(n-1,0,v[n-1]-360.0,v[0])}else{(n-1,0,v[n-1],v[0]+360.0)});
 let w2=(want-x)/(y-x);(a,b,1.0-w2,w2)
}
fn main()->io::Result<()>{
    // Keep the source stamp in the binary: #[used] does not stop MSVC's /OPT:REF
    // from dropping an unreferenced static (3.0.0 Windows cut: no stamp found).
    std::hint::black_box(SOURCE_REV);
    let args:Vec<String>=env::args().collect();if args.len()!=4{return Err(bad("usage: aerosol-grid PUBLIC_AEROSOL.dat grid-lat-lon-f32.bin output.bin"));}
 let mut r=BufReader::new(File::open(&args[1])?);let mut first=[0u8;4];r.read_exact(&mut first)?;let be=if i32::from_le_bytes(first)==12{false}else if i32::from_be_bytes(first)==12{true}else{return Err(bad("WRF aerosol dimension record must contain three int32 words"));};let mut dims=[0u8;12];r.read_exact(&mut dims)?;let mut marker=[0u8;4];r.read_exact(&mut marker)?;if marker!=first{return Err(bad("dimension record marker mismatch"));}
 let d:Vec<usize>=dims.chunks_exact(4).map(|v|if be{i32::from_be_bytes(v.try_into().unwrap())}else{i32::from_le_bytes(v.try_into().unwrap())}).map(|v|v as usize).collect();let (nx,ny,nz)=(d[0],d[1],d[2]);if nx<2||ny<2||nz!=30||nx>4096||ny>2048{return Err(bad("WRF aerosol grid needs bounded lon/lat dimensions and 30 aerosol levels"));}
 let lat=floats(record(&mut r,be)?,be,ny)?;let lon=floats(record(&mut r,be)?,be,nx)?;if lat.iter().chain(lon.iter()).any(|x|!x.is_finite()) || lat.windows(2).any(|x|x[1]<=x[0]) {return Err(bad("WRF aerosol latitude must be finite and increasing, and longitude must be finite"));}
 // The public WRF reader also accepts one periodic longitude wrap. Preserve
 // indices and its first bracket search; sorting changes interpolation words.
 let wraps=lon.windows(2).filter(|x|x[1]<x[0]).count();
 if wraps>1 || lon.windows(2).any(|x|x[1]==x[0] || (x[1]<x[0] && x[1]+360.0<=x[0])) {return Err(bad("WRF aerosol longitude needs increasing segments with at most one periodic wrap"));}
 let levels=nx*ny*nz*12;let pressure=floats(record(&mut r,be)?,be,levels)?;let aerosol=floats(record(&mut r,be)?,be,levels*10)?;
 let grid=std::fs::read(&args[2])?;if grid.is_empty()||grid.len()%8!=0{return Err(bad("target grid must contain equal nonempty lat and lon float32 arrays"));}let nc=grid.len()/8;let coords=floats(grid,false,nc*2)?;if coords.iter().any(|x|!x.is_finite()){return Err(bad("target coordinates must be finite"));}
 let mut weights=vec![((0usize,0usize,0f32,0f32),(0usize,0usize,0f32,0f32));nc];
 parallel_fill(&mut weights,|c,w|*w=(interp(&lat,coords[c]),interp(&lon,coords[nc+c])));
 let mut w=BufWriter::new(File::create(&args[3])?);
 // One species buffer bounds staging memory at 12*30*nc*4 bytes. Each worker
 // writes disjoint words; the native expression association is unchanged.
 let mut output=vec![[0u8;4];12*30*nc];
 for field in 0..11 {
  let data=if field==0{&pressure}else{&aerosol};let species=if field==0{0}else{field-1};
  parallel_fill(&mut output,|idx,word|{
   let column=idx%nc;let plane=idx/nc;let k=plane%30;let m=plane/30;
   let ((j1,j2,a,b),(i1,i2,c,d))=weights[column];
   let base=nx*ny*(k+30*(m+12*species));
   let x=c*(a*data[base+i1+nx*j1]+b*data[base+i1+nx*j2])+d*(a*data[base+i2+nx*j1]+b*data[base+i2+nx*j2]);
   *word=x.to_le_bytes();
  });
  w.write_all(output.as_flattened())?;
 }
 w.flush()?;println!("WRF aerosol grid {} {} {} -> {} columns, 10 species, 12 months, {} workers",nx,ny,nz,nc,workers());Ok(())}
