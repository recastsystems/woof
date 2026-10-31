//! Small numerical port fixture. Weather proof uses the public radar door.
use bowecho_simradar::wrf_band_scattering::{self as scattering,Cell};
use serde_json::json;
fn main() -> Result<(),String> {
    let mut rows=Vec::new();
    for band in ["C","X"] {
        for scheme in [8,9,18,50,51] {
            for (label,cell) in [
                ("rain",Cell {mass:[0.001,0.,0.,0.,0.],number:[1e5,0.,0.,0.,0.],density:1.,temperature:285.,..Cell::default()}),
                ("ice",Cell {mass:[0.,0.0003,0.0001,0.0002,0.0001],number:[0.,1e4,1e5,2e3,1e2],density:0.8,temperature:260.,volume:[5e-7,2e-7],rime:5e-5,rime_volume:1e-7}),
                ("melting",Cell {mass:[0.001,0.0003,0.0001,0.0002,0.0001],number:[1e5,1e4,1e5,2e3,1e2],density:1.,temperature:275.,volume:[5e-7,2e-7],rime:5e-5,rime_volume:1e-7}),
            ] {
                let value=scattering::integrate(scattering::table(band)?,cell,scheme,scattering::p3_table())?;
                let moments:Vec<_>=value.moments.iter().map(|m|[m.zh,m.zv,m.cov_re,m.cov_im,m.kdp_deg_km,m.ah_db_km,m.av_db_km]).collect();
                rows.push(json!({"band":band,"scheme":scheme,"label":label,"mass":cell.mass,"number":cell.number,
                    "density":cell.density,"temperature":cell.temperature,"volume":cell.volume,"rime":cell.rime,"rime_volume":cell.rime_volume,
                    "moments":moments,"mass_error":value.mass_error,"clipped_mass_fraction":value.clipped_mass_fraction}));
            }
        }
    }
    println!("{}",serde_json::to_string_pretty(&rows).map_err(|e|e.to_string())?);
    Ok(())
}
