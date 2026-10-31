//! RRFS fork src/core_atmosphere/physics/mpas_atmphys_initialize_real.F:200-1048
//! at 1d6e370e. Shared NOAH and monthly arithmetic are retained; RUC and ice
//! dispatch reproduce this fork's indexes, including its bucket assignment.
use crate::error::{MpasError,MpasResult};
use rayon::prelude::*;
pub use crate::init::soil::{DeepMoisture,SoilCounts,SeaiceCounts};
pub type SoilLayers=(Vec<Vec<f32>>,Vec<Vec<f32>>,Vec<Vec<f32>>,Vec<Vec<f32>>,SoilCounts);
/// RRFS fork src/core_atmosphere/physics/mpas_atmphys_initialize_real.F:282-381 at 1d6e370e.
pub fn layer_geometry(n:usize)->MpasResult<(Vec<f32>,Vec<f32>)> {
    match n {
        4=>Ok((crate::init::soil::noah_mid_depths().to_vec(),crate::init::soil::NOAH_LAYER_THICKNESS.to_vec())),
        9=>{let zs=vec![0.,0.01,0.04,0.10,0.30,0.60,1.,1.60,3.];let mut zb=vec![0.;9];let mut dz=vec![0.;9];zb[1]=(zs[1]+zs[0])*0.5;dz[0]=zb[1]-zb[0];for i in 1..8 {zb[i]=(zs[i+1]+zs[i])*0.5;dz[i]=zb[i]-zb[i-1];}zb[8]=zs[8];dz[8]=zb[8]-zb[7];Ok((zs,dz))},
        // The native select-case has no default arm: other layer counts keep
        // the pool's cold arrays (fork initialize_real.F:327-378).
        _=>Ok((vec![0.;n],vec![0.;n]))
    }
}
/// RRFS fork src/core_atmosphere/physics/mpas_atmphys_initialize_real.F:384-849 at 1d6e370e.
#[allow(clippy::too_many_arguments)]
fn soil_chunk(land:&[i32],skin:&[f32],tmn:&[f32],st:&[Vec<f32>],sm:&[Vec<f32>],dz_fg:&[f32],zs_fg:&[f32],n:usize,deep:DeepMoisture)->MpasResult<SoilLayers> {
    if n==4 {let(a,b,c,d,e)=crate::init::soil::init_soil_layers_properties(land,skin,tmn,st,sm,dz_fg,deep)?;let cv=|x:Vec<[f32;4]>|x.into_iter().map(|v|v.to_vec()).collect();return Ok((cv(a),cv(b),cv(c),cv(d),e));}
    let(zs,dz)=layer_geometry(n)?;let nf=dz_fg.len();let mut count=SoilCounts::default();
    for c in 0..land.len() {for k in 0..nf {if st[c][k]<=0. {count.bad_first_guess_temperature+=1;}if sm[c][k]<0. {count.bad_first_guess_moisture+=1;}}}
    if count.bad_first_guess_temperature!=0 || count.bad_first_guess_moisture!=0 {return Err(MpasError::Refusal(format!("fork soil fatal check: {} nonpositive temperatures, {} negative moisture values",count.bad_first_guess_temperature,count.bad_first_guess_moisture)));}
    let mut ts=vec![vec![0.;n];land.len()];let mut ms=ts.clone();let liq=ts.clone();let rel=ts.clone();
    if n!=9 {return Ok((ts,ms,liq,rel,count));}
    match nf {
        4=>{for c in 0..land.len() {
            let mut ti=vec![0.;6];let mut mi=vec![0.;6];let mut zh=vec![0.;6];ti[0]=skin[c];ti[5]=tmn[c];
            for k in 0..4 {ti[k+1]=st[c][k];mi[k+1]=sm[c][k];zh[k+1]=zs_fg[k]/100.;}
            let bucket=((mi[1]*0.1+mi[2]*0.2)+mi[3]*0.7)+mi[4]*1.;
            mi[0]=(mi[1]-mi[2])/(zs_fg[1]-zs_fg[0])*zs_fg[0]+mi[1];mi[5]=mi[4];zh[3]=300./100.;
            // Every defined target is bracketed before the native uninitialized
            // tail. Retain cold output if an input does not provide a bracket.
            for i in 0..n {for k in 0..5 {if zs[i]>=zh[k] && zs[i]<=zh[k+1] {ts[c][i]=(ti[k]*(zh[k+1]-zs[i])+ti[k+1]*(zs[i]-zh[k]))/(zh[k+1]-zh[k]);ms[c][i]=(mi[k]*(zh[k+1]-zs[i])+mi[k+1]*(zs[i]-zh[k]))/(zh[k+1]-zh[k]);break;}}}
            let mut ruc_bucket=0.;for i in 0..n-1 {ruc_bucket=bucket+ms[c][i]*dz[i];}
            if land[c]>0 {for i in 0..n-1 {let temp=ms[c][i]*bucket;let temp=temp/(0.9*ruc_bucket);ms[c][i]=0.02f32.max(temp);}
                let mut factor=vec![1.;n];if ms[c][1]>ms[c][0] && ms[c][2]>ms[c][1] {factor[..5].copy_from_slice(&[0.75,0.8,0.85,0.9,0.95]);}for i in 0..n-1 {ms[c][i]=factor[i]*ms[c][i];}
            }
        }},
        9=>{for c in 0..land.len() {ts[c].copy_from_slice(&st[c]);ms[c].copy_from_slice(&sm[c]);}},
        _=>{} // Native dispatcher leaves the cold arrays for other FG counts.
    }
    for c in 0..land.len() {if land[c]<1 {count.water_cells_filled+=1;ts[c].fill(skin[c]);ms[c].fill(1.);}}
    Ok((ts,ms,liq,rel,count))
}
/// Surface cleanup policy from the producer recipe.
#[derive(Debug, Clone, Copy, Default, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SeaiceCleanup {
    #[default]
    KeepInterpolated,
    ClearUnfrozen,
}

/// Fork initialize_real.F:918-1048 at the original pin, and :1046-1060
/// at 44c0849. Each cell is independent, including its soil rewrites.
#[allow(clippy::too_many_arguments)]
pub fn physics_init_seaice(frac:bool,threshold_skin:f32,cleanup:SeaiceCleanup,ice_lu:i32,land:&[i32],skin:&[f32],ice:&mut [f32],sea:&mut [f32],snow:&mut [f32],snowc:&mut [f32],snowh:&mut [f32],veg:&mut [f32],alb:&mut [f32],lu:&mut [i32],soil:&mut [i32],xland:&mut [f32],tmn:&mut [f32],ts:&mut [Vec<f32>],moist:&mut [Vec<f32>],liq:&mut [Vec<f32>],rel:&mut [Vec<f32>])->SeaiceCounts {
    let threshold=if frac {0.02} else {0.5};
    let (converted,zeroed,cleared) = (ice,sea,snow,snowc,snowh,veg,alb,lu,soil).into_par_iter()
        .zip((xland,tmn,ts,moist,liq,rel).into_par_iter()).enumerate()
        .map(|(c,((ice,sea,snow,snowc,snowh,veg,alb,lu,soil),(xland,tmn,ts,moist,liq,rel)))| {
            if !frac {*ice=if *ice>=0.5 {1.} else {0.};}
            // Native seaice starts from the case-7 >=0.5 flag. Every icy
            // cell is subsequently selected by the <=0.5 conversion threshold.
            *sea=0.;
            if *ice>=threshold || (land[c]==0 && skin[c]<threshold_skin) {
                if land[c]==0 {*tmn=271.4;}
                *lu=ice_lu; *soil=16; *alb=0.75; *veg=0.; *xland=1.; *sea=1.;
                *snowh=snowh.min(1.2); *snow=snow.min(360.);
                let n=ts.len();
                for i in 0..n {
                    let mid=3./n as f32/2.+i as f32*(3./n as f32);
                    ts[i]=((3.-mid)*skin[c]+mid* *tmn)/3.;
                    moist[i]=1.; liq[i]=0.; rel[i]=0.;
                }
                (1usize,0usize,0usize)
            } else if *ice<threshold {
                let zeroed=usize::from(land[c]==0 && *ice>0.);
                let cleared=usize::from(land[c]==0 && *snow>0.);
                if matches!(cleanup,SeaiceCleanup::ClearUnfrozen) {
                    *ice=0.;
                    if land[c]==0 {*snowc=0.; *snowh=0.; *snow=0.;}
                }
                if matches!(cleanup,SeaiceCleanup::ClearUnfrozen) {(0,zeroed,cleared)} else {(0,0,0)}
            } else {
                (0,0,0)
            }
        }).reduce(||(0,0,0),|(a,b,c),(d,e,f)|(a+d,b+e,c+f));
    SeaiceCounts {converted_to_land:converted,subthreshold_ice_zeroed:zeroed,
                  snow_cleared_over_water:cleared,xice_threshold:threshold}
}

// Preserve the existing per-column arithmetic while distributing every soil
// stage across the same rayon pool as the atmospheric stages.
fn grain(n:usize)->usize {(n/rayon::current_num_threads()).max(1)}

pub fn first_guess_consistency_pass(land:&[i32],st:&[Vec<f32>],sm:&mut [Vec<f32>])->(usize,usize) {
    let size=grain(land.len());
    sm.par_chunks_mut(size).enumerate().map(|(j,chunk)| {
        let a=j*size; let b=a+chunk.len();
        crate::init::soil::first_guess_consistency_pass(&land[a..b],&st[a..b],chunk)
    }).reduce(||(0,0),|(a,b),(c,d)|(a+c,b+d))
}

pub fn adjust_input_soiltemps(land:&[i32],ter:&[f32],soiltemp:&[f32],soilz:&[f32],skin:&mut [f32],st:&mut [Vec<f32>])->Vec<f32> {
    let size=grain(land.len());
    skin.par_chunks_mut(size).zip(st.par_chunks_mut(size)).enumerate().map(|(j,(skin,st))| {
        let a=j*size; let b=a+skin.len();
        crate::init::soil::adjust_input_soiltemps(&land[a..b],&ter[a..b],&soiltemp[a..b],&soilz[a..b],skin,st)
    }).collect::<Vec<_>>().into_iter().flatten().collect()
}

#[allow(clippy::too_many_arguments)]
pub fn init_soil_layers_properties(land:&[i32],skin:&[f32],tmn:&[f32],st:&[Vec<f32>],sm:&[Vec<f32>],dz_fg:&[f32],zs_fg:&[f32],n:usize,deep:DeepMoisture)->MpasResult<SoilLayers> {
    let size=grain(land.len());
    let chunks:Vec<_>=land.par_chunks(size).enumerate().map(|(j,chunk)| {
        let a=j*size; let b=a+chunk.len();
        soil_chunk(chunk,&skin[a..b],&tmn[a..b],&st[a..b],&sm[a..b],dz_fg,zs_fg,n,deep)
    }).collect();
    let(mut ts,mut ms,mut liq,mut rel)=(Vec::new(),Vec::new(),Vec::new(),Vec::new());
    let mut counts=SoilCounts::default();
    for chunk in chunks {
        let(mut a,mut b,mut c,mut d,e)=chunk?;
        ts.append(&mut a);ms.append(&mut b);liq.append(&mut c);rel.append(&mut d);
        counts.bad_first_guess_temperature+=e.bad_first_guess_temperature;
        counts.bad_first_guess_moisture+=e.bad_first_guess_moisture;
        counts.water_cells_filled+=e.water_cells_filled;
        counts.dry_floor_applied+=e.dry_floor_applied;
        counts.unbracketed_layers+=e.unbracketed_layers;
        counts.deep_moisture_choice_differs+=e.deep_moisture_choice_differs;
        counts.deep_moisture_max_difference=counts.deep_moisture_max_difference.max(e.deep_moisture_max_difference);
        counts.land_moisture_repaired+=e.land_moisture_repaired;
        counts.land_temperature_flagged+=e.land_temperature_flagged;
        counts.deep_moisture_anchor_reached+=e.deep_moisture_anchor_reached;
        counts.ruc_layer_floor_values+=e.ruc_layer_floor_values;
        counts.ruc_layer_floor_columns+=e.ruc_layer_floor_columns;
        counts.ruc_layer_floor_min=match (counts.ruc_layer_floor_min,e.ruc_layer_floor_min) {
            (Some(a),Some(b))=>Some(a.min(b)),(a,b)=>a.or(b),
        };
    }
    Ok((ts,ms,liq,rel,counts))
}

pub fn init_surface_companions(land:&[i32],snow:&[f32],albedo:&[Vec<f32>],green:&[Vec<f32>],year:i32,month:u32,day:u32)->crate::init::soil::SurfaceState {
    let mut background=crate::init::soil::monthly_interp_to_date(albedo,year,month,day);
    background.par_iter_mut().enumerate().for_each(|(i,v)| {*v= *v/100.;if land[i]==0 {*v=0.08;}});
    let vegetation=crate::init::soil::monthly_interp_to_date(green,year,month,day);
    let (minimum,maximum)=(0..land.len()).into_par_iter().map(|i| {
        let(mut lo,mut hi)=(green[0][i],green[0][i]);
        for m in 1..12 {if green[m][i]<lo {lo=green[m][i];}if green[m][i]>hi {hi=green[m][i];}}
        (lo,hi)
    }).unzip();
    crate::init::soil::SurfaceState {albedo:background,vegetation,minimum,maximum,
        snow_cover:snow.par_iter().map(|&v|if v>=10. {1.} else {0.}).collect(),
        snow_height:snow.par_iter().map(|&v|v*5./1000.).collect(),
        land:land.par_iter().map(|&v|if v==1 {1.} else if v==0 {2.} else {0.}).collect()}
}
