//! Complete native GWDO static module, in RKIND single precision.
//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:113-1449
//! at 1d6e370e. Fixed 30s tiles, USGS WATER=16 and in-place smoothing.
//! The ordinary archive-aware static_gwd module remains independent.
use std::{collections::BTreeMap,path::Path};
use crate::error::{MpasError,MpasResult};
use crate::static_gwd::GwdFields;

struct Tile {topo:Vec<i16>,landuse:Vec<i8>}
/// Native fixed signed big-endian tile decoder; geography index is unused.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:551-674 at 1d6e370e.
fn tile(topo:&Path,landuse:&Path,x:i32,y:i32,shift:i32)->MpasResult<Tile> {
    let tx=if x>shift {x-shift} else {x+shift};
    let filename=|x:i32|format!("{x:05}-{:05}.{y:05}-{:05}",x+1199,y+1199);
    let bytes=std::fs::read(topo.join(filename(tx)))?;
    let mask=std::fs::read(landuse.join(filename(x)))?;
    if bytes.len()!=1206*1206*2 || mask.len()!=1200*1200 {
        return Err(MpasError::Refusal("native GWD requires complete 1206x1206 signed-i16 terrain and 1200x1200 signed-i8 USGS tiles; a short fixture would leave native tile storage unset".into()));
    }
    let mut h=Vec::with_capacity(1200*1200);
    for j in 3..1203 {for i in 3..1203 {let n=(j*1206+i)*2;h.push(i16::from_be_bytes([bytes[n],bytes[n+1]]));}}
    Ok(Tile{topo:h,landuse:mask.into_iter().map(|v|v as i8).collect()})
}
/// Box gather, all ten reductions, then serial native isolated-point smoothing.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:113-385,704-1263 at 1d6e370e.
pub fn compute(topo:&Path,landuse:&Path,shift:i32,scaling:f32,
    lat:&[f64],lon:&[f64],dc:&[f64],counts:&[usize],neighbors:&[usize],max_edges:usize,
    progress:&dyn Fn(&str))->MpasResult<GwdFields> {
    let n=lat.len();let pi=std::f32::consts::PI;let re=6371229f32;let rad2deg=180f32/pi;
    let mut cache:BTreeMap<(i32,i32),(u64,Tile)>=BTreeMap::new();
    let mut tile_access=0u64;
    let mut result=GwdFields {var2d:vec![0.;n],con:vec![0.;n],oa:std::array::from_fn(|_|vec![0.;n]),
        ol:std::array::from_fn(|_|vec![0.;n]),hlanduse:vec![0;n],water_category:16,
        water_category_source:"native USGS WATER=16".into(),..Default::default()};
    for c in 0..n {
        let latitude=lat[c] as f32;let dx=dc[c] as f32*scaling;
        let nx=if latitude.cos()>(2f32*120f32*dx*180f32)/(43200f32*pi*re) {
            ((180f32*dx*120f32)/(pi*re*latitude.cos())).ceil() as i32
        } else {21600};
        let ny=((180f32*dx*120f32)/(pi*re)).ceil() as i32;
        let mut ic=((lon[c] as f32*rad2deg+180f32)*120f32).round() as i32+1;
        let jc=((latitude*rad2deg+90f32)*120f32).round() as i32+1;
        if ic<=0 {ic+=43200;}if ic>43200 {ic-=43200;}
        let mut box_h=Vec::with_capacity((nx*ny) as usize);let mut box_l=Vec::with_capacity((nx*ny) as usize);
        let mut mean=0f32;
        for j in 1..=ny {let mut i=1;while i<=nx {
            let mut ii=i-nx/2+ic;let mut jj=j-ny/2+jc;
            if jj<=0 {jj=-jj+1;ii+=21600;}if jj>21600 {jj=21600-(jj-21600-1);ii+=21600;}
            ii=(ii-1).rem_euclid(43200)+1;
            let x=(ii-1)/1200*1200+1;let y=(jj-1)/1200*1200+1;
            if !cache.contains_key(&(x,y)) {
                // Evict by use, not geographic key. Key-order eviction can
                // reread the same tile for every pixel of a polar box.
                if cache.len()>=32 {let oldest=*cache.iter().min_by_key(|(_,entry)|entry.0).unwrap().0;cache.remove(&oldest);}
                cache.insert((x,y),(tile_access,tile(topo,landuse,x,y,shift)?));
            }
            tile_access+=1;
            let entry=cache.get_mut(&(x,y)).unwrap();entry.0=tile_access;
            let t=&entry.1;let k=((jj-y)*1200+ii-x) as usize;
            let span=(x+1200-ii).min(nx-i+1) as usize;
            // Gather consecutive native i indices in their original order.
            // The reductions retain every RKIND addition and rounding.
            for offset in 0..span {let h=t.topo[k+offset] as f32;box_h.push(h);box_l.push(t.landuse[k+offset]);mean+=h;}
            i+=span as i32;
        }}
        let size=(nx*ny) as f32;mean/=size;
        let(mut xland,mut ml,mut mw)=(0f32,0f32,0f32);
        for (&h,&lu) in box_h.iter().zip(&box_l) {if lu!=16 {xland+=1f32;ml+=h;}else{mw+=h;}}
        if xland>0f32 {ml/=xland;}if xland<size {mw/=size-xland;}xland/=size;
        let oro=if xland>=0.5f32 {ml}else{mw};
        let(mut s2,mut s4)=(0f32,0f32);
        let mut up=[0i32;4];let mut down=[0i32;4];let ratio=ny as f32/nx as f32;
        for j in 1..=ny {for i in 1..=nx {
            let h=box_h[((j-1)*nx+i-1) as usize];let d=h-mean;let a=h-oro;let a2=a*a;
            s2+=d*d;s4+=a2*a2;
            if h>mean {
                for (k,u) in [i<=nx/2,j<=ny/2,((i as f32*ratio).round() as i32)<ny-j,
                    ((i as f32*ratio).round() as i32)<j].into_iter().enumerate() {
                    if u {up[k]+=1;}else{down[k]+=1;}
                }
            }
        }}
        let variance=s2/size;let sigma=variance.sqrt();result.var2d[c]=sigma as f64;
        result.con[c]=if sigma<1f32 || xland<0.5f32 {0.} else {(s4/(variance*variance*size)) as f64};
        for k in 0..4 {result.oa[k][c]=if up[k]+down[k]>0 {((up[k]-down[k]) as f32/(up[k]+down[k]) as f32) as f64}else{0.};}
        let hc=1116.2f32-0.878f32*sigma;let mut above=[0i32;4];let mut total=[0i32;4];
        for j in 1..=ny {for i in 1..=nx {
            let h=box_h[((j-1)*nx+i-1) as usize];
            for (k,chosen) in [j>=ny/4&&j<=3*ny/4,i>=nx/4&&i<=3*nx/4,
                (j<=ny/2&&i<=nx/2)||(j>ny/2&&i>nx/2),
                (j>ny/2&&i<=nx/2)||(j<=ny/2&&i>nx/2)].into_iter().enumerate() {
                if chosen {total[k]+=1;if h>hc {above[k]+=1;}}
            }
        }}
        for k in 0..4 {result.ol[k][c]=(above[k] as f32/total[k] as f32) as f64;}
        result.hlanduse[c]=u8::from(xland>=0.5f32);
        if c%128==0 {progress(&format!("GWD_NATIVE\t{c}\t{n}"));}
    }
    for c in 0..n {
        let(mut land,mut sum)=(0i8,0f32);
        for j in 0..counts[c] {
            let neighbor=neighbors[c*max_edges+j];
            if neighbor>=n {
                // RRFSx/MPAS-Model gwd.F:218,313-314 at 1d6e370e reads
                // hlanduse(nCells+1), which is allocated but not initialized.
                // Whole strict regional-stage repeats on both hosts, twice
                // at one/two ranks, reproduce zero contributions. Receipts:
                // native-gwd-regional-stage-node{1,2}-v{2,1}.json. This also
                // agrees with i3's identical-source native sentinel witness.
                sum+=0.0f32;
                continue;
            }
            land=land.wrapping_add(result.hlanduse[neighbor] as i8);sum+=result.var2d[neighbor] as f32;
        }
        if (land==counts[c] as i8&&result.hlanduse[c]==0)||(land==0&&result.hlanduse[c]==1) {
            result.var2d[c]=(sum/counts[c] as f32) as f64;result.smoothed_cells+=1;
        }
    }
    Ok(result)
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub input:std::path::PathBuf,pub output:std::path::PathBuf,
    pub terrain:std::path::PathBuf,pub landuse:std::path::PathBuf,
    pub topo_shift:i32,pub cell_scaling:f32,
}
/// Whole fixed-format GWD stage on an existing mesh/static carrier.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:113-357 at 1d6e370e.
pub fn run(r:Request)->MpasResult<crate::init::artifact::Receipt> {
    let file=netcrust::File::open(&r.input)?;
    let reals=|name:&str|->MpasResult<Vec<f64>> {Ok(file.read_array::<f32>(name)?.into_iter().map(|v|v as f64).collect())};
    let lat=reals("latCell")?;let lon=reals("lonCell")?;let edges=reals("dcEdge")?;
    let counts:Vec<usize>=file.read_array::<i32>("nEdgesOnCell")?.into_iter().map(|n|n as usize).collect();
    let eoc:Vec<usize>=file.read_array::<i32>("edgesOnCell")?.into_iter().map(|n|n.wrapping_sub(1) as usize).collect();
    let coc:Vec<usize>=file.read_array::<i32>("cellsOnCell")?.into_iter().map(|n|n.wrapping_sub(1) as usize).collect();
    let max_edges=file.dimension("maxEdges").unwrap().len();
    let radius=file.attribute("sphere_radius").and_then(|a|a.as_f64()).unwrap_or(6371229.);
    let unit=edges.iter().copied().fold(0f64,f64::max)<1.;
    let mut dc=Vec::with_capacity(lat.len());
    for (c,&count) in counts.iter().enumerate() {
        let mut mean=0f32;for j in 0..count {mean+=edges[eoc[c*max_edges+j]] as f32;}
        mean/=count as f32;if unit {mean*=radius as f32;}dc.push(mean as f64);
    }
    let fields=compute(&r.terrain,&r.landuse,r.topo_shift,r.cell_scaling,&lat,&lon,&dc,&counts,&coc,max_edges,&|_|{})?;
    let mut values=vec![("var2d".to_string(),fields.var2d),("con".to_string(),fields.con)];
    for (k,array) in fields.oa.into_iter().enumerate() {values.push((format!("oa{}",k+1),array));}
    for (k,array) in fields.ol.into_iter().enumerate() {values.push((format!("ol{}",k+1),array));}
    // The native GWD-only output stream contains static fields, even when its
    // input carrier includes meteorology. Preserve that stream inventory;
    // carrying inactive input fields would add unused vertical/soil dimensions.
    // RRFSx/MPAS-Model Registry.xml:1105-1149 at 1d6e370e.
    let selected=std::iter::once("Time")
        .chain(crate::rrfs_static::declared_variables().iter().copied())
        .filter(|name|!matches!(*name,"lu_index"|"soilcat_top"))
        .map(str::to_string).collect();
    crate::init::artifact::materialize_selected(crate::init::artifact::Request{overlays:Vec::new(),template:r.input,output:r.output,
        dimensions:BTreeMap::new(),attributes:BTreeMap::new(),variables:values.into_iter().map(|(name,values)|
            crate::init::artifact::Variable{name,dimensions:vec!["nCells".into()],attributes:BTreeMap::new(),
                values:Some(values.into_iter().map(|v|v as f32).collect()),text:None,only_if_absent:false,only_if_zero:false}).collect()},Some(&selected))
}

const PPD:f32=120.;
const RAD2DEG:f32=180./std::f32::consts::PI;
pub const SG_DELTA:f32=((2.*std::f32::consts::PI)*6371229.)/(360.*PPD);

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:1268-1449 at 1d6e370e.
/// Reused from i3 because the complete pinned source file is byte-identical.
/// Native's calls at :300-303 are commented out; these are not output fields.
#[derive(Debug, Clone, Copy)]
pub struct DormantDiagnostics {
    pub elvmax: f32,
    pub htheta: f32,
    pub hgamma: f32,
    pub hsigma: f32,
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:794-795 at 1d6e370e.
pub fn pixel_dx(jj: i64) -> f32 {
    let sg_lat = (-90.0 + ((jj - 1) as f32 + 0.5) / PPD) / RAD2DEG;
    SG_DELTA * sg_lat.cos()
}

/// Full dormant get_elvmax/get_htheta/get_hgamma/get_hsigma reductions.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_gwd.F:1268-1287,1300-1339,1352-1396 at 1d6e370e,
/// :1409-1449. Identical gradient sums are shared without reordering.
pub fn dormant_diagnostics(
    topo: &[f32],
    dxm: &[f32],
    nx: usize,
    ny: usize,
) -> MpasResult<DormantDiagnostics> {
    if topo.is_empty() || nx.checked_mul(ny) != Some(topo.len())
        || topo.len() != dxm.len()
    {
        return Err(MpasError::Refusal("native GWD gradient diagnostics require complete terrain and pixel-metric storage".into()));
    }
    let mut elvmax = topo[0];
    for &v in topo {
        if v > elvmax {
            elvmax = v;
        }
    }
    let (mut hx2, mut hy2, mut hxy) = (0.0f32, 0.0f32, 0.0f32);
    for j in 1..ny.saturating_sub(1) {
        for i in 1..nx.saturating_sub(1) {
            let k = j * nx + i;
            let xfp = (topo[k + 1] - topo[k - 1]) / (2.0 * dxm[k]);
            let yfp = (topo[k + nx] - topo[k - nx]) / (2.0 * SG_DELTA);
            hx2 += xfp * xfp;
            hy2 += yfp * yfp;
            hxy += xfp * yfp;
        }
    }
    let n = ((nx as i64 - 2) * (ny as i64 - 2)) as f32;
    hx2 /= n;
    hy2 /= n;
    hxy /= n;
    let hk = 0.5 * (hx2 + hy2);
    let hl = 0.5 * (hx2 - hy2);
    let hlp = (hl * hl + hxy * hxy).sqrt();
    Ok(DormantDiagnostics {
        elvmax,
        htheta: 0.5 * hxy.atan2(hl),
        hgamma: if hk + hlp > 0.0 && hk - hlp >= 0.0 {
            ((hk - hlp) / (hk + hlp)).sqrt()
        } else {
            0.0
        },
        hsigma: (hk + hlp).sqrt(),
    })
}

