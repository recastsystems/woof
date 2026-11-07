//! Public NASA MERRA-2 GMI monthly climatology -> WRF's ten-species ABI.
//! Written from papers, not bitwise to WRF. See docs/public-cumulus-aerosol.md.
//! No CESM source data, values, fitted corrections, or restricted tables used.

#[used]
static SOURCE_REV: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));
use std::{env, fs::{self, File}, io::{self, Write, BufWriter}, path::Path, process::Command, thread};

const BASE: &str = "https://opendap.nccs.nasa.gov/dods/merra2_gmi/instM_3d_aer_Nv";
const VARS: [&str; 17] = ["delp", "airdens", "du001", "du002", "du003", "du004", "du005", "ss001", "ss002", "ss003", "ss004", "ss005", "so4", "bcphobic", "bcphilic", "ocphobic", "ocphilic"];
// Retain every near-surface layer and progressively fewer stratospheric layers.
const LEVELS: [usize;30] = [0,6,12,18,24,28,32,36,40,42,44,46,48,50,52,54,56,58,60,61,62,63,64,65,66,67,68,69,70,71];
fn bad(s: impl Into<String>) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s.into()) }
fn workers() -> usize { env::var("GPUWM_AEROSOL_GRID_THREADS").ok().and_then(|v|v.parse().ok()).unwrap_or(8usize).clamp(2,32) }
fn fill<T:Send,F:Fn(usize,&mut T)+Sync>(out: &mut [T], f:F) {
    let chunk=out.len().div_ceil(workers());
    thread::scope(|scope| { for (i,part) in out.chunks_mut(chunk.max(1)).enumerate() {
        let f=&f; scope.spawn(move || {for (j,x) in part.iter_mut().enumerate() {f(i*chunk+j,x);}});
    }});
}
// DAP2 XDR: first Float32 grid array has two identical network-order counts.
// Coordinate maps follow it. Validate the DDS dimensions and both counts before
// decoding; never treat an HTML error, fill value, or truncated reply as data.
fn decode(bytes: &[u8], dims: &[usize], month: Option<usize>) -> io::Result<Vec<f32>> {
    let split=bytes.windows(6).position(|w|w==b"Data:\n").ok_or_else(||bad("NASA response lacks DAP2 Data delimiter"))?;
    let dds=std::str::from_utf8(&bytes[..split]).map_err(|_|bad("invalid DAP2 DDS"))?;
    for (name,size) in ["time","lev","lat","lon"].iter().zip(dims) {
        if !dds.contains(&format!("[{name} = {size}]")) {return Err(bad(format!("unexpected NASA {name} dimension: {dds}")));}
    }
    let n=dims.iter().product::<usize>(); let start=split+6;
    if bytes.len()<start+8+n*4 {return Err(bad("short DAP2 array"));}
    for offset in [0,4] {if u32::from_be_bytes(bytes[start+offset..start+offset+4].try_into().unwrap()) as usize!=n {return Err(bad("DAP2 array count differs from DDS"));}}
    let mut cursor=start+8+n*4;
    for (axis,&count) in dims.iter().enumerate() {
        if bytes.len()<cursor+8+count*8 {return Err(bad("short DAP2 coordinate map"));}
        for offset in [0,4] {
            if u32::from_be_bytes(bytes[cursor+offset..cursor+offset+4].try_into().unwrap()) as usize!=count {return Err(bad("DAP2 coordinate count differs from DDS"));}
        }
        cursor+=8;
        let mut previous=f64::NEG_INFINITY;
        for i in 0..count {
            let x=f64::from_be_bytes(bytes[cursor+i*8..cursor+(i+1)*8].try_into().unwrap());
            let expected=match axis {1=>Some((i+1) as f64),2=>Some(-90.+i as f64*180./(count-1) as f64),3=>Some(-180.+i as f64*360./count as f64),_=>None};
            if !x.is_finite() || x<=previous || expected.is_some_and(|want|(x-want).abs()>1e-8) {return Err(bad("NASA coordinate grid differs from declared source"));}
            previous=x;
        }
        cursor+=count*8;
    }
    if cursor!=bytes.len() {return Err(bad("unexpected trailing DAP2 data"));}
    // Annual replies are reused for twelve output months. Decode only the
    // selected month; all DDS/count/map checks still run on each reply, and
    // every value is validated when its month is consumed.
    let (first,count)=match month {
        Some(m) if m<dims[0] => (m*(n/dims[0]),n/dims[0]),
        Some(_) => return Err(bad("month outside NASA reply")),
        None => (0,n),
    };
    let mut out=vec![0f32;count];
    fill(&mut out,|i,x|*x=f32::from_be_bytes(bytes[start+8+(first+i)*4..start+12+(first+i)*4].try_into().unwrap()));
    if out.iter().any(|x|!x.is_finite() || *x<0. || *x>=1e14) {return Err(bad("NASA array has missing, negative, or nonfinite values"));}
    Ok(out)
}
fn download_year(cache:&Path,name:&str,year:usize,stride:usize,ny:usize,nx:usize)->io::Result<()> {
    let file=cache.join(format!("{name}-y{year}-s{stride}.dods"));
    if file.is_file() { return Ok(()); }
    let start=(year-1980)*12;
    let url=format!("{BASE}.dods?{name}[{start}:{}][0:71][0:{stride}:360][0:{stride}:575]",start+11);
    let temp=file.with_extension(format!("part-{}",std::process::id()));
    eprintln!("fetch {url}");
    let status=Command::new("curl").args(["--globoff","--fail","--location","--silent","--show-error","--retry","3","--retry-all-errors","--connect-timeout","30","--max-time","900","--output"]).arg(&temp).arg(&url).status()?;
    if !status.success() {let _=fs::remove_file(temp);return Err(bad(format!("NASA download failed: {name}, year {year}")));}
    let bytes=fs::read(&temp)?; decode(&bytes,&[12,72,ny,nx],None)?;
    fs::rename(temp,file)?; Ok(())
}
fn fetch(cache:&Path,name:&str,month:usize,stride:usize,ny:usize,nx:usize)->io::Result<Vec<f32>> {
    let count=72*ny*nx;let mut out=Vec::with_capacity(10*count);
    for year in 2001..=2010 {
        let file=cache.join(format!("{name}-y{year}-s{stride}.dods"));
        let selected=decode(&fs::read(file)?,&[12,72,ny,nx],Some(month))?;
        out.extend_from_slice(&selected);
    }
    Ok(out)
}
// Mass-conserving overlap in log dry diameter. This within-bin assumption is
// explicit, since neither monthly source nor the paper resolves sub-bin mass.
fn overlap(lo:f64,hi:f64,a:f64,b:f64)->f64 {
    let left=lo.max(a);let right=hi.min(b);
    if right<=left {0.} else {(right/left).ln()/(hi/lo).ln()}
}
fn record(w:&mut impl Write, bytes:&[u8])->io::Result<()> {
    let marker=i32::try_from(bytes.len()).map_err(|_|bad("Fortran record exceeds int32"))?.to_le_bytes();
    w.write_all(&marker)?;w.write_all(bytes)?;w.write_all(&marker)
}
fn main()->io::Result<()> {
    // Keep the source stamp in the binary: #[used] does not stop MSVC's /OPT:REF
    // from dropping an unreferenced static (3.0.0 Windows cut: no stamp found).
    std::hint::black_box(SOURCE_REV);
    let args:Vec<_>=env::args().collect();
    if args.len()<3 || args.len()>4 {return Err(bad("usage: public-aerosol-build CACHE OUTPUT.dat [NASA-grid-stride=2]"));}
    let stride=args.get(3).map(|s|s.parse::<usize>()).transpose().map_err(|_|bad("invalid stride"))?.unwrap_or(2);
    if ![1,2,4,8].contains(&stride) {return Err(bad("stride must be 1, 2, 4 or 8 (full NASA grid or declared subsampling)"));}
    let cache=Path::new(&args[1]);fs::create_dir_all(cache)?;
    let (nx,ny)=(575/stride+1,360/stride+1);let nc=nx*ny;
    // Contiguous annual requests avoid the server timing out on decade-wide
    // strided queries. Four independent downloads, at most eight CPU workers
    // decoding each bounded response. Raw responses are reusable offline.
    let jobs:Vec<_>=(2001..=2010).flat_map(|y|VARS.iter().map(move|n|(y,*n))).collect();
    for batch in jobs.chunks(4) {
        thread::scope(|scope| {
            let tasks:Vec<_>=batch.iter().map(|&(y,n)|scope.spawn(move||download_year(cache,n,y,stride,ny,nx))).collect();
            for task in tasks {task.join().unwrap()?;} Ok::<(),io::Error>(())
        })?;
    }
    let plane=30*nc;let mut pressure=vec![[0u8;4];12*plane];
    let mut aerosol=vec![[0u8;4];10*12*plane];
    let dust_in=[0.2f64,2.,3.6,6.,12.,20.];let dust_out=[0.1f64,1.,2.5,5.,10.];
    let mut weights=[[0f64;5];4];for d in 0..4 {for s in 0..5 {weights[d][s]=overlap(dust_in[s],dust_in[s+1],dust_out[d],dust_out[d+1]);}}
    for month in 0..12 {
        // Two concurrent cached-file decodes; no exclusive CPU allocation.
        let mut data=Vec::new();
        for names in VARS.chunks(2) {
            let result=thread::scope(|scope| {
                let tasks:Vec<_>=names.iter().map(|name|scope.spawn(move||fetch(cache,name,month,stride,ny,nx))).collect();
                tasks.into_iter().map(|h|h.join().unwrap()).collect::<io::Result<Vec<_>>>()
            })?;
            data.extend(result);
        }
        let mut p=vec![0f64;10*72*nc];
        // Each task owns an entire vertical column; 1 Pa is GEOS model top.
        let mut columns=vec![[0f64;72];10*nc];
        fill(&mut columns,|i,col| {let (year,c)=(i/nc,i%nc);let mut edge=1f64;for (k,x) in col.iter_mut().enumerate() {
            let dp=data[0][(year*72+k)*nc+c] as f64;*x=edge+dp*0.5;edge+=dp;
        }});
        fill(&mut p,|i,x| {let c=i%nc;let k=i/nc%72;let year=i/(nc*72);*x=columns[year*nc+c][k];});
        if columns.iter().any(|c|c.windows(2).any(|w|w[1]<=w[0])) {return Err(bad("nonmonotonic NASA layer pressure"));}
        fill(&mut pressure[month*plane..(month+1)*plane],|i,x| {
            let (k,c)=(LEVELS[i/nc],i%nc);let value=(0..10).map(|y|p[(y*72+k)*nc+c]).sum::<f64>()/10.;*x=(value as f32).to_le_bytes();
        });
        for species in 0..10 {
            let offset=(species*12+month)*plane;
            fill(&mut aerosol[offset..offset+plane],|i,x| {
                let (k,c)=(LEVELS[i/nc],i%nc);let mut mean=0f64;
                for y in 0..10 {let n=(y*72+k)*nc+c;
                    let mmr=match species {
                        0..=3=>(0..5).map(|b|weights[species][b]*data[2+b][n] as f64).sum(),
                        4=>(7..12).map(|b|data[b][n] as f64).sum(),
                        // Sulfate is the SO4 ion, not ammonium sulfate. Organic
                        // fields are already particulate matter; no extra 1.4.
                        _=>data[species+7][n] as f64,
                    };
                    mean+=mmr*data[1][n] as f64*1e9/10.;
                }
                *x=(mean as f32).to_le_bytes();
            });
        }
        eprintln!("converted month {}, {} x {} columns, 30 levels, {} workers",month+1,nx,ny,workers());
    }
    let output=Path::new(&args[2]);let temp=output.with_extension(format!("part-{}",std::process::id()));
    let mut w=BufWriter::new(File::create(&temp)?);
    let dims=[nx as i32,ny as i32,30i32].map(i32::to_le_bytes);record(&mut w,dims.as_flattened())?;
    let lat:Vec<_>=(0..ny).map(|j|(-90f32+j as f32*stride as f32*0.5).to_le_bytes()).collect();
    let lon:Vec<_>=(0..nx).map(|i|(-180f32+i as f32*stride as f32*0.625).to_le_bytes()).collect();
    record(&mut w,lat.as_flattened())?;record(&mut w,lon.as_flattened())?;
    record(&mut w,pressure.as_flattened())?;record(&mut w,aerosol.as_flattened())?;
    w.flush()?;drop(w);fs::rename(temp,output)?;
    println!("written from papers, not bitwise to WRF; NASA MERRA-2 GMI 2001-2010; grid {nx} {ny} 30; ug m-3; pressure Pa");
    Ok(())
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn dust_mapping_and_excluded_coarse_tail_conserve_each_input_bin() {
        let src=[0.2,2.,3.6,6.,12.,20.];let dst=[0.1,1.,2.5,5.,10.];
        for s in 0..5 {let sum=(0..4).map(|d|overlap(src[s],src[s+1],dst[d],dst[d+1])).sum::<f64>()+overlap(src[s],src[s+1],10.,20.);assert!((sum-1.).abs()<1e-14);}
    }
    #[test] fn reject_incomplete_or_wrong_dimension_dap() {
        assert!(decode(b"<html>error</html>",&[10,72,46,72],None).is_err());
        assert!(decode(b"Dataset {[time = 1][lev = 1][lat = 1][lon = 1]}\nData:\n",&[1,1,1,1],None).is_err());
    }
}
