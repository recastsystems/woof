//! Calls production case-7 routines for the pristine-Fortran replay harness.
//! Real native boundary captures and complete pristine-module replay; forecasts are visual evidence.
use rw_mpas::init::{soil, vinterp, hinterp, dynamics, boundary, metfile, advection};
use serde::Deserialize;
use std::{fs, io::Write};

#[derive(Deserialize)]
#[serde(tag = "mode")]
enum Request {
    DecodeMet {met:String},
    NativeGridGeometry {grid:String,reference:String},
    Sorted {coordinates:Vec<f32>,values:Vec<f32>,draws:Vec<f32>},
    Horizontal {met:String,lat:Vec<f32>,lon:Vec<f32>,elat:Vec<f32>,elon:Vec<f32>,land:Vec<i32>,nfg:usize,nsoil:usize,#[serde(default)] prepared:bool},
    Order2Scalar {reference:String},
    FieldWords {files:Vec<String>,field:String},
    RealOperators {mesh:String,reference:String},
    RealGeometryVertical {reference:String,#[serde(default)] terrain_reference:Option<String>},
    ExtractRealMet {met:String},
    SurfaceInput {met:String,lat:Vec<f32>,lon:Vec<f32>,land:Vec<i32>,sst:Vec<f32>,skin:Vec<f32>,ice:Vec<f32>,frac:bool,interpolate:bool},
    Auxiliary {met:String,kind:String,lat:Vec<f32>,lon:Vec<f32>,land:Vec<i32>,initial:Vec<f32>},
    Reconstruction {u:Vec<Vec<f32>>,counts:Vec<usize>,edges:Vec<Vec<usize>>,coefficients:Vec<f32>,lat:Vec<f32>,lon:Vec<f32>,sphere:bool},
    Surface {land:Vec<i32>,snow:Vec<f32>,albedo:Vec<Vec<f32>>,green:Vec<Vec<f32>>,year:i32,month:u32,day:u32},
    Winds { u:Vec<Vec<f32>>,rho:Vec<Vec<f32>>,owners:Vec<[usize;2]>,counts:Vec<usize>,edges:Vec<Vec<usize>>,
        zz:Vec<Vec<f32>>,fzm:Vec<f32>,fzp:Vec<f32>,zb:Vec<[Vec<f32>;2]>,zb3:Vec<[Vec<f32>;2]>,order:i32,coefficient:f32 },
    Operators { fixture:advection::Request },
    Column { t:Vec<f32>, pressure:Vec<f32>, rh:Vec<f32>, sh:Vec<f32>, use_sh:bool,
        zgrid:Vec<f32>, zz:Vec<f32>, fzm:Vec<f32>, fzp:Vec<f32>, dzu:Vec<f32>, rdzw:f32,
        t2m:f32, psfc:f32, rh2:f32 },
    Boundary { nx:usize,ny:usize,values:Vec<f32>,start_lat:f32,start_lon:f32,
        delta_lat:f32,delta_lon:f32,lat:Vec<f32>,lon:Vec<f32>,mask:Vec<i32>,terrain:Vec<f32> },
    Soil { land: Vec<i32>, terrain: Vec<f32>, soiltemp: Vec<f32>, soilz: Vec<f32>,
        skin: Vec<f32>, st: Vec<Vec<f32>>, sm: Vec<Vec<f32>>, dz: Vec<f32>,
        frac: bool, threshold: f32, ice: Vec<f32>, #[serde(default)] missing_depth:bool },
    Vinterp { coordinates: Vec<f32>, values: Vec<f32>, targets: Vec<f32>, extrap: String },
    Oned { points: Vec<[f32; 5]> },
    Hinterp { nx: usize, ny: usize, values: Vec<f32>, mask: Vec<f32>,
        maskval: Option<f32>, points: Vec<[f32;2]>, methods: Vec<i32>, msgval: f32 },
}

#[test]
fn native_replay() {
    let Ok(input) = std::env::var("CASE7_REQUEST") else { return; };
    let mut out = fs::File::create(std::env::var("CASE7_OUTPUT").unwrap()).unwrap();
    let bytes=fs::read(input).unwrap();
    let payload=if bytes.starts_with(b"ORCL0001") {f1_payload(&bytes,"pre","request")} else {bytes};
    let request: Request = serde_json::from_slice(&payload).unwrap();
    fn array(out: &mut impl Write, values: &[f32]) {
        for v in values { out.write_all(&v.to_le_bytes()).unwrap(); }
    }
    match request {
        Request::DecodeMet {met}=> {
            for field in metfile::read_met_file(std::path::Path::new(&met)).unwrap() {
                let native_proj=match field.iproj {1=>3,3=>1,5=>2,n=>n};
                for v in [field.format_version,native_proj,field.nx as i32,field.ny as i32,field.wind_grid_rel as i32] {out.write_all(&v.to_le_bytes()).unwrap();}
                for (text,width) in [(&field.hdate,24),(&field.map_source,32),(&field.field,9),(&field.units,25),(&field.desc,46)] {
                    let mut bytes=text.as_bytes().to_vec();bytes.resize(width,b' ');out.write_all(&bytes).unwrap();
                }
                array(&mut out,&[field.xfcst,field.xlvl,field.start_i,field.start_j,field.start_lat,field.start_lon,field.earth_radius_km]);
                if field.iproj==0 || field.iproj==4 {array(&mut out,&[field.delta_lat,field.delta_lon]);}
                else {array(&mut out,&[field.dx,field.dy,field.truelat1]);if field.iproj!=1 {array(&mut out,&[field.stdlon]);}if field.iproj==3 {array(&mut out,&[field.truelat2]);}}
                array(&mut out,&field.values);
            }
        }
        Request::NativeGridGeometry {grid,reference}=> {
            let raw=netcrust::File::open(&grid).unwrap();let native=netcrust::File::open(&reference).unwrap();
            let radius=native.attribute("sphere_radius").unwrap().as_f64().unwrap() as f32;
            let mut rows=Vec::new();
            for name in ["xCell","yCell","zCell","xEdge","yEdge","zEdge","xVertex","yVertex","zVertex","dcEdge","dvEdge","nominalMinDc","areaCell","areaTriangle","kiteAreasOnVertex"] {
                let input=raw.read_array_f64(name).unwrap().into_values();
                let values=advection::scale_native_grid_field(name,&input,radius).unwrap();
                let expected:Vec<_>=native.read_array_f64(name).unwrap().into_values().into_iter().map(|v|v as f32).collect();
                assert_eq!(values.len(),expected.len());
                let differing:Vec<_>=values.iter().zip(&expected).enumerate().filter_map(|(i,(a,b))|(a.to_bits()!=b.to_bits()).then_some(i)).collect();
                rows.push(serde_json::json!({"field":name,"words":values.len(),"differing_words":differing.len(),"first_indices":differing.iter().take(8).collect::<Vec<_>>()}));
            }
            let same=rows.iter().all(|r|r["differing_words"]==0);
            out.write_all(&serde_json::to_vec_pretty(&serde_json::json!({"schema":"mpas.native-grid-scale/v1","radius":radius,"grid":grid,"reference":reference,"all_identical":same,"fields":rows})).unwrap()).unwrap();
            assert!(same,"native input RKIND geometry scaling differs");
        }
        Request::Sorted {mut coordinates,mut values,draws}=> {
            let mut draws=draws.into_iter();
            vinterp::quicksort_column_with(&mut coordinates,&mut values,||draws.next().expect("native pivot trace exhausted"));
            assert!(draws.next().is_none(),"port consumed fewer draws than native");
            array(&mut out,&coordinates);array(&mut out,&values);
        }
        Request::Horizontal {met,lat,lon,elat,elon,land,nfg,nsoil,prepared}=> {
            use rw_mpas::init::{capsule::{MeshGeometry,Statics},InitReceipt,horizontal_pass};
            let nc=lat.len();let ne=elat.len();
            let mesh=MeshGeometry {n_cells:nc,n_edges:ne,lat_cell:lat,lon_cell:lon,lat_edge:elat,lon_edge:elon,
                angle_edge:vec![0.;ne],cells_on_edge:vec![[1,2];ne],n_edges_on_cell:vec![1;nc],edges_on_cell:vec![vec![1];nc]};
            let statics=Statics {landmask:land,ter:vec![0.;nc],soiltemp:vec![280.;nc],ivgtyp:vec![1;nc],isltyp:vec![1;nc],snoalb:vec![0.;nc],isice_lu:24,greenfrac:vec![],albedo12m:vec![]};
            let slabs=metfile::read_met_file(std::path::Path::new(&met)).unwrap();let mut receipt=InitReceipt::default();
            let mut fg=horizontal_pass(&slabs,&mesh,&statics,nfg,nsoil,hinterp::Underflow::Preserve,false,&mut receipt).unwrap();
            let surface=prepared.then(||rw_mpas::init::prepare_vertical_pass(&mut fg));
            let fg=serde_json::to_value(fg).unwrap();
            for name in ["vert_level","t","rh","sh","z","p","u","v","psfc","pmsl","soilz","skintemp","snow","xice","u10","v10","st","sm","dzs_cm"] {
                for value in fg[name].as_array().unwrap() {
                    if let Some(row)=value.as_array() {for v in row {array(&mut out,&[v.as_f64().unwrap() as f32]);}}
                    else {array(&mut out,&[value.as_f64().unwrap() as f32]);}
                }
            }
            if let Some((t,rh,q))=surface {array(&mut out,&t);array(&mut out,&rh);array(&mut out,&q);}
        }
        Request::Order2Scalar {reference}=> {
            // MPAS-A v8.4.2 mpas_init_atm_cases.F:3407-3438 leaves
            // z_edge3 unset for order2. Identify the actual strict-binary
            // value by reproducing both rounded operations on every word.
            let native=netcrust::File::open(&reference).unwrap();
            let read=|name:&str|native.read_array_f64(name).unwrap().into_values().into_iter().map(|v|v as f32).collect::<Vec<_>>();
            let zb3=read("zb3");let dv=read("dvEdge");let area=read("areaCell");
            let owners:Vec<usize>=read("cellsOnEdge").into_iter().map(|v|v as usize-1).collect();
            let nzp=zb3.len()/(dv.len()*2);let estimate=(zb3[0]*area[owners[0]])/dv[0];
            let mut candidates=Vec::new();
            for bits in estimate.to_bits().saturating_sub(32)..=estimate.to_bits().saturating_add(32) {
                let scalar=f32::from_bits(bits);let mut differing=0;
                for e in 0..dv.len() {for side in 0..2 {let value=scalar*dv[e]/area[owners[e*2+side]];
                    for k in 0..nzp-1 {if value.to_bits()!=zb3[(e*2+side)*nzp+k].to_bits() {differing+=1;}}
                }}
                if differing==0 {candidates.push(serde_json::json!({"scalar_bits":bits,"scalar":scalar}));}
            }
            let nc=area.len();let max=native.dimension("maxEdges").unwrap().len();
            let indices=|name:&str|read(name).into_iter().map(|v|v as i64-1).collect::<Vec<_>>();
            let z=read("zgrid");
            let rows=|name:&str|indices(name).chunks_exact(max).map(|v|v.to_vec()).collect::<Vec<_>>();
            let request=serde_json::json!({"mode":"edge","width":4,"terrain":read("ter"),
                "counts":read("nEdgesOnCell").into_iter().map(|v|v as usize).collect::<Vec<_>>(),
                "edges_on_cell":rows("edgesOnCell"),"cells_on_cell":rows("cellsOnCell"),
                "cells_on_edge":indices("cellsOnEdge").chunks_exact(2).map(|v|[v[0],v[1]]).collect::<Vec<_>>(),
                "dc":read("dcEdge"),"dv":dv,"area":area,"levels":nzp-1,"top":30000.,"scheme":"tc","specified":null,
                "projection":"linear_interpolation","terrain_passes":0,"smooth":true,"surface_passes":30,
                "minimum_fraction":0.3,"hybrid":true,"transition":30000.,"damping":0.,"damping_start":22000.,"regional_halo":false,
                "order":2,"deriv":[],"zgrid":(0..nzp).map(|k|(0..nc).map(|c|z[c*nzp+k]).collect::<Vec<_>>()).collect::<Vec<_>>()});
            let mut bytes=Vec::new();rw_mpas::init::vertical::run(serde_json::to_vec(&request).unwrap().as_slice(),&mut bytes).unwrap();
            let words:Vec<u32>=bytes[zb3.len()*4..].chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
            let mut differing=0;
            for e in 0..dv.len() {for side in 0..2 {for k in 0..nzp {
                if words[(k*2+side)*dv.len()+e]!=zb3[(e*2+side)*nzp+k].to_bits() {differing+=1;}
            }}}
            out.write_all(&serde_json::to_vec_pretty(&serde_json::json!({"schema":"mpas.native-order2-scalar/v1","reference":reference,"active_words":dv.len()*2*(nzp-1),"estimate":estimate,"matching_scalars":candidates,"production_words":zb3.len(),"production_differing_words":differing})).unwrap()).unwrap();
            assert!(!candidates.is_empty(),"no constant scalar reproduces strict order2 words");
            assert_eq!(differing,0,"production order2 edge words differ from native");
        }
        Request::FieldWords {files,field}=> {
            // Measure stored native words without widening floats, so NaN
            // payloads and signed zeros remain part of the defect witness.
            use netcdf_reader::{NcFormat, classic::{header::parse_header,data::{compute_record_stride,read_record_variable,read_non_record_variable}}};
            use sha2::{Digest,Sha256};
            let mut results=Vec::new();let mut baseline:Option<Vec<u32>>=None;
            for path in files {
                let bytes=fs::read(&path).unwrap();
                let format=match bytes.get(..4) {Some(b"CDF\x01")=>NcFormat::Classic,Some(b"CDF\x02")=>NcFormat::Offset64,Some(b"CDF\x05")=>NcFormat::Cdf5,_=>panic!("native witness must be a classic netCDF file")};
                let header=parse_header(&bytes,format).unwrap();
                let variable=header.variables.iter().find(|v|v.name==field).unwrap();
                let values=if variable.dimensions.iter().any(|d|d.is_unlimited) {read_record_variable::<f32>(&bytes,variable,header.numrecs,compute_record_stride(&header.variables).unwrap())}
                    else {read_non_record_variable::<f32>(&bytes,variable)}.unwrap();
                let bits:Vec<u32>=values.into_iter().map(f32::to_bits).collect();let mut digest=Sha256::new();
                let mut counts=std::collections::BTreeMap::<u32,usize>::new();
                for &b in &bits {digest.update(b.to_le_bytes());*counts.entry(b).or_default()+=1;}
                let differences:Vec<usize>=baseline.as_ref().map(|first|{assert_eq!(first.len(),bits.len());first.iter().zip(&bits).enumerate().filter_map(|(i,(a,b))|(a!=b).then_some(i)).collect()}).unwrap_or_default();
                let mut ranges=Vec::<[usize;2]>::new();
                for i in &differences {if let Some(last)=ranges.last_mut().filter(|v|v[1]+1==*i) {last[1]=*i;} else {ranges.push([*i,*i]);}}
                results.push(serde_json::json!({"file":path,"words":bits.len(),"sha256_le_words":format!("{:x}",digest.finalize()),
                    "nonzero_words":bits.iter().filter(|&&b|b&0x7fffffff!=0).count(),"nonfinite_words":bits.iter().filter(|&&b|b&0x7f800000==0x7f800000).count(),
                    "distinct_word_values":counts.len(),"value_counts":counts.iter().take(16).map(|(b,n)|serde_json::json!({"bits":b,"words":n})).collect::<Vec<_>>(),
                    "differing_words_from_first":differences.len(),"differing_index_ranges_inclusive":ranges}));
                if baseline.is_none() {baseline=Some(bits);}
            }
            out.write_all(&serde_json::to_vec_pretty(&serde_json::json!({"schema":"mpas.native-field-word-witness/v1","field":field,"runs":results})).unwrap()).unwrap();
        }
        Request::RealOperators {mesh,reference}=> {
            // MPAS-A v8.4.2 core_atmosphere/mpas_atm_advection.F:21-923.
            // Decode the real mesh in Rust and exercise the production producer;
            // carried native coefficients cannot satisfy this comparison.
            use rw_mpas::static_operators::{OperatorMesh,build_operator_fields};
            let file=netcrust::File::open(&mesh).unwrap();
            let read=|name:&str|file.read_array_f64(name).unwrap().into_values();
            let dim=|name:&str|file.dimension(name).unwrap().len();
            let index=|name:&str|read(name).into_iter().map(|v|if v<=0. {usize::MAX} else {v as usize-1}).collect::<Vec<_>>();
            let nc=dim("nCells");let ne=dim("nEdges");let max=dim("maxEdges");
            let counts:Vec<usize>=read("nEdgesOnCell").into_iter().map(|v|v as usize).collect();
            let neighbors=index("cellsOnCell");let edges=index("edgesOnCell");let vertices=index("verticesOnCell");
            let owners:Vec<[usize;2]>=index("cellsOnEdge").chunks_exact(2).map(|v|[v[0],v[1]]).collect();
            let edge_vertices:Vec<[usize;2]>=index("verticesOnEdge").chunks_exact(2).map(|v|[v[0],v[1]]).collect();
            let coords:Vec<Vec<f64>>=["xCell","yCell","zCell"].map(read).into();
            let vcoords:Vec<Vec<f64>>=["xVertex","yVertex","zVertex"].map(read).into();
            let angles=read("angleEdge");let dc=read("dcEdge");
            let radius=file.attribute("sphere_radius").unwrap().as_f64().unwrap();
            let mesh=OperatorMesh {n_cells:nc,n_edges:ne,max_edges:max,sphere_radius:radius,on_a_sphere:true,
                n_edges_on_cell:&counts,cells_on_cell:&neighbors,edges_on_cell:&edges,cells_on_edge:&owners,
                vertices_on_cell:&vertices,vertices_on_edge:&edge_vertices,
                x_cell:&coords[0],y_cell:&coords[1],z_cell:&coords[2],
                x_vertex:&vcoords[0],y_vertex:&vcoords[1],z_vertex:&vcoords[2],angle_edge:&angles,dc_edge:&dc};
            let fields=build_operator_fields(&mesh).unwrap();
            let native=netcrust::File::open(&reference).unwrap();let mut results=Vec::new();
            for (name,values) in [("deriv_two",fields.deriv_two),("cell_gradient_coef_x",fields.cell_gradient_coef_x),("cell_gradient_coef_y",fields.cell_gradient_coef_y)] {
                let expected:Vec<f32>=native.read_array_f64(name).unwrap().into_values().into_iter().map(|v|v as f32).collect();
                assert_eq!(values.len(),expected.len(),"{name} shape");
                let differing:Vec<usize>=values.iter().zip(&expected).enumerate().filter_map(|(i,(a,b))|(a.to_bits()!=b.to_bits()).then_some(i)).collect();
                let examples:Vec<_>=differing.iter().take(8).map(|&i|serde_json::json!({"index":i,"native_bits":expected[i].to_bits(),"rust_bits":values[i].to_bits()})).collect();
                results.push(serde_json::json!({"field":name,"words":values.len(),"differing_words":differing.len(),"examples":examples}));
            }
            let identical=results.iter().all(|v|v["differing_words"]==0);
            out.write_all(&serde_json::to_vec_pretty(&serde_json::json!({"schema":"mpas.case7.real-operators/v1","n_cells":nc,"n_edges":ne,"max_edges":max,"reference":reference,"all_identical":identical,"fields":results})).unwrap()).unwrap();
            assert!(identical,"real mesh operator words differ");
        }
        Request::RealGeometryVertical {reference,terrain_reference}=> {
            // Native boundary replay of complete vector/RBF and coordinate producers.
            // MPAS v8.4.2 operators/mpas_vector_operations.F:652-771,
            // operators/mpas_vector_reconstruction.F:61-191,
            // core_init_atmosphere/mpas_init_atm_cases.F:3097-3443.
            // A native static-stage capture supplies terrain before case7 smoothing.
            // Older callers may explicitly use the post-smoothing boundary.
            let file=netcrust::File::open(&reference).unwrap();
            let read=|name:&str|file.read_array_f64(name).unwrap().into_values();
            let dim=|name:&str|file.dimension(name).unwrap().len();
            let nc=dim("nCells");let ne=dim("nEdges");let me=dim("maxEdges");let nz=dim("nVertLevels");
            let index=|name:&str|read(name).into_iter().map(|v|v as i64-1).collect::<Vec<_>>();
            let mut results=Vec::new();
            let mut compare=|name:&str,values:Vec<f32>| {
                let expected=read(name).into_iter().map(|v|v as f32).collect::<Vec<_>>();
                assert_eq!(values.len(),expected.len(),"{name} shape");
                let differing=values.iter().zip(&expected).enumerate().filter_map(|(i,(a,b))|(a.to_bits()!=b.to_bits()).then_some(i)).collect::<Vec<_>>();
                results.push(serde_json::json!({"field":name,"words":values.len(),"differing_words":differing.len(),
                    "examples":differing.iter().take(8).map(|&i|serde_json::json!({"index":i,"native_bits":expected[i].to_bits(),"rust_bits":values[i].to_bits()})).collect::<Vec<_>>()}));
            };
            let mut input=Vec::new();input.extend_from_slice(b"HEXGEO1\0");
            for size in [nc,ne,me,nc] {input.extend_from_slice(&(size as u64).to_le_bytes());}
            input.extend_from_slice(&[4,1,0,0]);input.extend_from_slice(&[0;8]);
            for (names,count) in [(["xCell","yCell","zCell"],nc),(["xEdge","yEdge","zEdge"],ne)] {
                let xyz=names.map(read);
                for i in 0..count {for axis in &xyz {input.extend_from_slice(&(axis[i] as f32).to_le_bytes());}}
            }
            for name in ["cellsOnEdge","edgesOnCell"] {for value in index(name) {input.extend_from_slice(&value.to_le_bytes());}}
            for value in read("nEdgesOnCell") {input.extend_from_slice(&(value as i64).to_le_bytes());}
            let mut output=Vec::new();rw_mpas::geometry::run(input.as_slice(),&mut output).unwrap();
            let words=output.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            let mut offset=0;
            for (name,size) in [("edgeNormalVectors",ne*3),("localVerticalUnitVectors",nc*3),("cellTangentPlane",nc*6),("coeffs_reconstruct",nc*me*3)] {
                compare(name,words[offset..offset+size].to_vec());offset+=size;
            }
            assert_eq!(offset,words.len());
            let rows=|name:&str|index(name).chunks_exact(me).map(|v|v.to_vec()).collect::<Vec<_>>();
            let deriv=read("deriv_two");let stencil=deriv.len()/(ne*2);
            let zgrid=read("zgrid");
            let terrain=terrain_reference.as_ref().map(|path|netcrust::File::open(path).unwrap().read_array_f64("ter").unwrap().into_values()).unwrap_or_else(||read("ter"));
            let terrain_passes=usize::from(terrain_reference.is_some());
            let mut request=serde_json::json!({"mode":"grid","width":4,"terrain":terrain,
                "counts":read("nEdgesOnCell").iter().map(|v|*v as usize).collect::<Vec<_>>(),
                "edges_on_cell":rows("edgesOnCell"),"cells_on_cell":rows("cellsOnCell"),
                "cells_on_edge":index("cellsOnEdge").chunks_exact(2).map(|v|[v[0],v[1]]).collect::<Vec<_>>(),
                "dc":read("dcEdge"),"dv":read("dvEdge"),"area":read("areaCell"),"levels":nz,
                "top":30000.,"scheme":"tc","specified":null,"projection":"linear_interpolation",
                "terrain_passes":terrain_passes,"smooth":true,"surface_passes":30,"minimum_fraction":0.3,
                "hybrid":true,"transition":30000.,"damping":0.,"damping_start":22000.,
                "regional_halo":false,"order":3,
                "deriv":deriv.chunks_exact(2*stencil).map(|v|vec![v[..stencil].to_vec(),v[stencil..].to_vec()]).collect::<Vec<_>>(),
                "zgrid":(0..nz+1).map(|k|(0..nc).map(|c|zgrid[c*(nz+1)+k]).collect::<Vec<_>>()).collect::<Vec<_>>()});
            request["mode"]=serde_json::json!("terrain");
            output.clear();rw_mpas::init::vertical::run(serde_json::to_vec(&request).unwrap().as_slice(),&mut output).unwrap();
            compare("ter",output.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect());
            request["mode"]=serde_json::json!("grid");
            output.clear();rw_mpas::init::vertical::run(serde_json::to_vec(&request).unwrap().as_slice(),&mut output).unwrap();
            let words=output[..output.len()-8].chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            offset=0;
            for (name,size,stored) in [("zw",nz+1,false),("dzw",nz,false),("rdzw",nz,true),("zu",nz,false),("dzu",nz,true),("rdzu",nz,true),("rdzwp",nz,false),("rdzwm",nz,false),("fzp",nz,true),("fzm",nz,true),("ah",nz+1,false)] {
                if stored {compare(name,words[offset..offset+size].to_vec());}offset+=size;
            }
            offset+=nc*(nz+1); // hx is an internal intermediate, absent from the native stream.
            for (name,levels,count) in [("zgrid",nz+1,nc),("zz",nz,nc),("zxu",nz,ne),("dss",nz,nc)] {
                let mut values=Vec::with_capacity(count*levels);
                for c in 0..count {for k in 0..levels {values.push(words[offset+k*count+c]);}}
                compare(name,values);offset+=levels*count;
            }
            for name in ["cf1","cf2","cf3"] {compare(name,vec![words[offset]]);offset+=1;}
            assert_eq!(offset,words.len());
            request["mode"]=serde_json::json!("edge");output.clear();
            rw_mpas::init::vertical::run(serde_json::to_vec(&request).unwrap().as_slice(),&mut output).unwrap();
            let words=output.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();offset=0;
            for name in ["zb","zb3"] {
                let mut values=Vec::with_capacity(ne*2*(nz+1));
                for e in 0..ne {for side in 0..2 {for k in 0..nz+1 {values.push(words[offset+(k*2+side)*ne+e]);}}}
                compare(name,values);offset+=ne*2*(nz+1);
            }
            assert_eq!(offset,words.len());
            let identical=results.iter().all(|r|r["differing_words"]==0);
            out.write_all(&serde_json::to_vec_pretty(&serde_json::json!({"schema":"mpas.case7.real-geometry-vertical/v1",
                "reference":reference,"process_count":1,"terrain_reference":terrain_reference,"terrain_smoothing_passes":terrain_passes,
                "scope":"all stored vector/RBF, terrain and coordinate outputs; internal intermediates separately compared against standalone Fortran",
                "all_identical":identical,"fields":results})).unwrap()).unwrap();
            assert!(identical,"real geometry/vertical output words differ");
        }
        Request::ExtractRealMet {met}=> {
            // Real field decoding and column preparation stay in the rustwx crates.
            let slabs=metfile::read_met_file(std::path::Path::new(&met)).unwrap();
            let heights:Vec<_>=slabs.iter().filter(|s|s.field=="GHT" && s.xlvl!=vinterp::SURFACE_LEVEL_TAG).collect();
            let land=slabs.iter().find(|s|s.field=="LANDSEA").unwrap();
            let mut requests=Vec::new();
            for point in 0..24 {
                let i=1+(point*37)%(land.nx-8);let j=1+(point*23)%(land.ny-8);
                for name in ["TT","RH","UU","VV"] {
                    let mut coordinates=Vec::new();let mut values=Vec::new();
                    for h in &heights {
                        if let Some(s)=slabs.iter().find(|s|s.field==name && s.xlvl==h.xlvl) {
                            coordinates.push(h.at(i,j));values.push(s.at(i,j));
                        }
                    }
                    vinterp::sorted_column(&mut coordinates,&mut values);
                    if coordinates.len()<2 {continue;}
                    let targets:Vec<f32>=std::iter::once(coordinates[0]-500.)
                        .chain(coordinates.iter().copied()).chain(coordinates.windows(2).map(|v|0.5*(v[0]+v[1])))
                        .chain(std::iter::once(coordinates[coordinates.len()-1]+500.)).collect();
                    for mode in ["constant","linear","lapse-rate"] {
                        requests.push(serde_json::json!({"mode":"Vinterp","field":name,"point":[i,j],
                            "coordinates":coordinates,"values":values,"targets":targets,"extrap":mode}));
                    }
                }
                // Real coastal mask and field windows, not a Python decoder.
                let sample=|name:&str,level:Option<f32>|slabs.iter()
                    .find(|s|s.field==name && level.is_none_or(|v|s.xlvl==v)).map(|s|s.at(i,j));
                let st:Option<Vec<f32>>=["ST000010","ST010040","ST040100","ST100200"].into_iter()
                    .map(|name|sample(name,None)).collect();
                let sm:Option<Vec<f32>>=["SM000010","SM010040","SM040100","SM100200"].into_iter()
                    .map(|name|sample(name,None)).collect();
                if let (Some(st),Some(sm),Some(skin),Some(soilz))=(st,sm,sample("SKINTEMP",None),sample("SOILHGT",None)) {
                    for frac in [false,true] {
                        if st.iter().any(|v|*v<=0.) || sm.iter().any(|v|*v<0.) {continue;}
                        requests.push(serde_json::json!({"mode":"Soil","field":"soil","point":[i,j],
                            "land":[land.at(i,j) as i32],"terrain":[soilz],"soilz":[soilz],"soiltemp":[st[3]],
                            "skin":[skin],"st":[st],"sm":[sm],"dz":[10.,30.,60.,100.],"frac":frac,
                            "threshold":271.4,"ice":[sample("SEAICE",None).unwrap_or(0.)]}));
                    }
                }
                let mut coordinates:Vec<f32>=heights.iter().map(|h|h.at(i,j)).collect();
                let mut positions:Vec<f32>=(0..heights.len()).map(|v|v as f32).collect();
                vinterp::sorted_column(&mut coordinates,&mut positions);
                if coordinates.len()>2 && coordinates.windows(2).all(|v|v[1]>v[0]) {
                    let nz=coordinates.len()-1;
                    let targets:Vec<f32>=coordinates.windows(2).map(|v|0.5*(v[0]+v[1])).collect();
                    let interpolate=|name:&str,log:bool| {
                        let values:Option<Vec<f32>>=positions.iter().map(|&index|sample(name,Some(heights[index as usize].xlvl)))
                            .map(|v|v.map(|v|if log {v.ln()} else {v})).collect();
                        values.map(|values|targets.iter().map(|&z| {
                            let v=vinterp::vertical_interp(z,&coordinates,&values,vinterp::Extrap::Linear).unwrap();
                            if log {v.exp()} else {v}
                        }).collect::<Vec<f32>>())
                    };
                    if let (Some(t),Some(rh))=(interpolate("TT",false),interpolate("RH",false)) {
                        let logp:Vec<f32>=positions.iter().map(|&n|heights[n as usize].xlvl.ln()).collect();
                        let pressure:Vec<f32>=targets.iter().map(|&z|vinterp::vertical_interp(z,&coordinates,&logp,vinterp::Extrap::Linear).unwrap().exp()).collect();
                        let dzu:Vec<f32>=targets.windows(2).map(|v|v[1]-v[0]).chain(std::iter::once(targets[nz-1]-targets[nz-2])).collect();
                        for use_sh in [false,true] {
                            requests.push(serde_json::json!({"mode":"Column","field":"column","point":[i,j],
                                "t":t,"rh":rh,"pressure":pressure,"sh":vec![0f32;nz],"use_sh":use_sh,
                                "zgrid":coordinates,"zz":vec![1f32;nz],"fzm":vec![0.5f32;nz],"fzp":vec![0.5f32;nz],
                                "dzu":dzu,"rdzw":1f32/(coordinates[1]-coordinates[0]),
                                "t2m":sample("TT",Some(vinterp::SURFACE_LEVEL_TAG)).unwrap(),
                                "psfc":sample("PSFC",None).unwrap(),"rh2":sample("RH",Some(vinterp::SURFACE_LEVEL_TAG)).unwrap()}));
                        }
                    }
                }
                for name in ["TT","SNOW","ST000010","SM000010","SEAICE"] {
                    let Some(s)=slabs.iter().find(|s|s.field==name) else {continue;};
                    let mut values=Vec::new();let mut mask=Vec::new();
                    for y in j..j+8 {for x in i..i+8 {values.push(s.at(x,y));mask.push(land.at(x,y));}}
                    for methods in [vec![1],vec![2],vec![3],vec![4],vec![5],vec![6],vec![7],vec![8],vec![1,2,5,7,4,6,3,8]] {
                        requests.push(serde_json::json!({"mode":"Hinterp","field":name,"point":[i,j],
                            "nx":8,"ny":8,"values":values,"mask":mask,"maskval":0.,
                            "points":[[1.,1.],[3.5,3.5],[7.125,2.33],[0.5,8.49]],"methods":methods,"msgval":-1e30f32}));
                    }
                }
            }
            out.write_all(&serde_json::to_vec(&requests).unwrap()).unwrap();
        }
        Request::SurfaceInput {met,lat,lon,land,mut sst,mut skin,mut ice,frac,interpolate}=> {
            if interpolate {
                let slabs=metfile::read_met_file(std::path::Path::new(&met)).unwrap();
                rw_mpas::init::surface::interpolate(&slabs,&lat,&lon,&land,&mut sst,&mut ice,hinterp::Underflow::Preserve).unwrap();
            }
            rw_mpas::init::surface::initialize_sst(frac,&land,&sst,&mut skin,&mut ice);
            for v in [&sst,&ice,&skin] {array(&mut out,v);}
        }
        Request::Auxiliary {met,kind,lat,lon,land,mut initial}=> {
            let slabs=metfile::read_met_file(std::path::Path::new(&met)).unwrap();
            let kind=match kind.as_str() {"ice"=>rw_mpas::init::auxiliary::Kind::SeaIce,"ocean"=>rw_mpas::init::auxiliary::Kind::OceanDepth,_=>panic!("unknown auxiliary kind")};
            rw_mpas::init::auxiliary::interpolate(&slabs,kind,&lat,&lon,&land,&mut initial,hinterp::Underflow::Preserve).unwrap();
            array(&mut out,&initial);
            let s=&slabs[0];
            let projection=match kind {rw_mpas::init::auxiliary::Kind::SeaIce=>rw_mpas::init::projection::Projection::auxiliary_polar(s).unwrap(),
                rw_mpas::init::auxiliary::Kind::OceanDepth=>rw_mpas::init::projection::Projection::from_met(s).unwrap()};
            let conversion=180f32/std::f32::consts::PI;
            for c in 0..land.len() {let (x,y)=if land[c]==1 {(0.,0.)} else {projection.locate(lat[c]*conversion,lon[c]*conversion)};array(&mut out,&[x,y]);}
        }
        Request::Reconstruction {u,counts,edges,coefficients,lat,lon,sphere}=> {
            let state=rw_mpas::init::reconstruction::reconstruct(&u,&counts,&edges,&coefficients,&lat,&lon,sphere).unwrap();
            for values in state.fields {array(&mut out,&values);}
        }
        Request::Surface {land,snow,albedo,green,year,month,day}=> {
            let state=soil::init_surface_companions(&land,&snow,&albedo,&green,year,month,day);
            for v in [&state.albedo,&state.vegetation,&state.minimum,&state.maximum,
                &state.snow_cover,&state.snow_height,&state.land] {array(&mut out,v);}
        }

        Request::Winds {u,rho,owners,counts,edges,zz,fzm,fzp,zb,zb3,order,coefficient} => {
            let nz=fzm.len();let metrics=rw_mpas::init::capsule::VerticalMetrics {n_vert_levels:nz,
                zgrid:vec![],zz,fzm,fzp,dzu:vec![],rdzw:vec![],zb,zb3};
            let rho:Vec<&[f32]>=rho.iter().map(|v|v.as_slice()).collect();
            let state=dynamics::diagnose_winds(&u,&rho,&owners,&counts,&edges,&metrics,order,coefficient);
            for v in [&state.ru,&state.rw,&state.w] {for row in v {array(&mut out,row);}}
        }
        Request::Operators {fixture:r} => {
            use rw_mpas::static_operators::{OperatorMesh,build_operator_fields};
            let nc=r.cells.len();let ne=r.owners.len();let max=r.counts.iter().copied().max().unwrap();
            let neighbors:Vec<usize>=r.neighbors.iter().flatten().map(|&c|if c<0 {usize::MAX} else {c as usize}).collect();
            let edges:Vec<usize>=r.edges.iter().flatten().copied().collect();
            let owners:Vec<[usize;2]>=r.owners.iter().map(|v|v.map(|c|if c<0 {usize::MAX} else {c as usize})).collect();
            let vertices:Vec<usize>=r.cell_vertices.iter().flatten().copied().collect();
            let coords:Vec<Vec<f64>>=(0..3).map(|i|r.cells.iter().map(|v|v[i]).collect()).collect();
            let vcoords:Vec<Vec<f64>>=(0..3).map(|i|r.vertices.iter().map(|v|v[i]).collect()).collect();
            let mesh=OperatorMesh {n_cells:nc,n_edges:ne,max_edges:max,sphere_radius:r.radius,on_a_sphere:r.sphere,
                n_edges_on_cell:&r.counts,cells_on_cell:&neighbors,edges_on_cell:&edges,cells_on_edge:&owners,
                vertices_on_cell:&vertices,vertices_on_edge:&r.edge_vertices,
                x_cell:&coords[0],y_cell:&coords[1],z_cell:&coords[2],
                x_vertex:&vcoords[0],y_vertex:&vcoords[1],z_vertex:&vcoords[2],
                angle_edge:&r.angles,dc_edge:&r.dc};
            let f=build_operator_fields(&mesh).unwrap();
            for v in [&f.deriv_two,&f.cell_gradient_coef_x,&f.cell_gradient_coef_y] {array(&mut out,v);}
        }
        Request::Column { t,pressure,mut rh,sh,use_sh,zgrid,zz,fzm,fzp,dzu,rdzw,t2m,psfc,rh2 } => {
            let state=dynamics::build_column(&t,&pressure,&rh,&sh,
                use_sh && sh.iter().any(|&v|v!=0.),&zgrid,&zz,&fzm,&fzp,&dzu,rdzw,
                dynamics::VirtualFactor::ReproduceFortran);
            dynamics::convert_relhum_wrt_ice(&t,&mut rh);
            for v in [&state.qv,&state.theta,&state.theta_m,&state.pressure,&state.pressure_base,
                &state.pressure_p,&state.exner,&state.exner_base,&state.rho_zz,&state.rho,
                &state.rho_base,&state.rho_p,&state.theta_base,&state.rtheta_base] {array(&mut out,v);}
            array(&mut out,&[state.precipw,state.surface_pressure]);
            array(&mut out,&rh);array(&mut out,&[dynamics::diagnose_q2(t2m,psfc,rh2)]);
        }
        Request::Boundary { nx,ny,values,start_lat,start_lon,delta_lat,delta_lon,lat,lon,mask,mut terrain } => {
            let field=metfile::MetSlab { format_version:5,xfcst:0.,start_i:1.,start_j:1.,wind_grid_rel:false,dx:0.,dy:0.,stdlon:0.,truelat1:0.,truelat2:0.,iproj:0, hdate:String::new(),map_source:String::new(),field:"SOILHGT".into(),
                units:"m".into(),desc:String::new(),xlvl:200100.,nx,ny,start_lat,start_lon,delta_lat,delta_lon,
                earth_radius_km:6371.,polar:None,values };
            boundary::blend_bdy_terrain(&field,&lat,&lon,&mask,&mut terrain).unwrap();
            array(&mut out,&terrain);
        }
        Request::Soil { land, terrain, soiltemp, soilz, mut skin, mut st, mut sm, mut dz,
            frac, threshold, mut ice, missing_depth } => {
            let n = land.len();
            if missing_depth {dz[0]=f32::NAN;}
            let (flagged,repaired) = soil::first_guess_consistency_pass(&land,&st,&mut sm);
            let mut tmn = soil::adjust_input_soiltemps(&land,&terrain,&soiltemp,&soilz,&mut skin,&mut st);
            let (mut tslb,mut smois,mut sh2o,mut smcrel,_) = soil::init_soil_layers_properties(
                &land,&skin,&tmn,&st,&sm,&dz,soil::DeepMoisture::ReproduceFortran).unwrap();
            let mut seaice = vec![0.0;n]; let mut snow = vec![12.3;n];
            let mut snowc = vec![1.0;n]; let mut snowh = vec![0.021;n];
            let mut vegfra = vec![73.2;n]; let mut snoalb = vec![0.81;n];
            let mut ivgtyp = vec![3;n]; let mut isltyp = vec![7;n];
            let mut xland:Vec<f32> = land.iter().map(|&v| 2.0-v as f32).collect();
            soil::physics_init_seaice(frac,threshold,24,&land,&skin,&mut ice,&mut seaice,
                &mut snow,&mut snowc,&mut snowh,&mut vegfra,&mut snoalb,&mut ivgtyp,
                &mut isltyp,&mut xland,&mut tmn,&mut tslb,&mut smois,&mut sh2o,&mut smcrel);
            for v in [&skin,&tmn,&ice,&seaice,&snow,&snowc,&snowh,&vegfra,&snoalb,&xland] { array(&mut out,v); }
            for v in [&st,&sm] { for row in v { array(&mut out,row); } }
            for v in [&tslb,&smois,&sh2o,&smcrel] { for row in v { array(&mut out,row); } }
            for v in [&ivgtyp,&isltyp] { for x in v { out.write_all(&x.to_le_bytes()).unwrap(); } }
            for v in [flagged,repaired] { out.write_all(&(v as i32).to_le_bytes()).unwrap(); }
        }
        Request::Vinterp { coordinates,values,targets,extrap } => {
            let mode = vinterp::Extrap::parse(&extrap).unwrap();
            for target in targets {
                match vinterp::vertical_interp(target,&coordinates,&values,mode) {
                    Ok(v) => { out.write_all(&0i32.to_le_bytes()).unwrap(); array(&mut out,&[v]); }
                    Err(_) => { out.write_all(&1i32.to_le_bytes()).unwrap(); array(&mut out,&[0.0]); }
                }
            }
        }
        Request::Oned { points } => {
            // MPAS v8.4.2 mpas_init_atm_hinterp.F:1025-1057.
            let slab = hinterp::Slab::new(1, 1, 1, 1);
            let ctx = hinterp::InterpContext { array: &slab, msgval: -1.0e30,
                maskval: None, mask: None, underflow: hinterp::Underflow::Preserve,
                underflow_hits: None };
            for [x,a,b,c,d] in points {
                array(&mut out, &[hinterp::oned(x,a,b,c,d,&ctx)]);
            }
        }
        Request::Hinterp { nx,ny,values,mask,maskval,points,methods,msgval } => {
            let slab = hinterp::Slab::from_met(&values,nx,ny);
            let mask = hinterp::Slab::from_met(&mask,nx,ny);
            let ctx = hinterp::InterpContext { array:&slab,msgval,maskval,mask:Some(&mask),
                underflow:hinterp::Underflow::Preserve,underflow_hits:None };
            let methods:Vec<_> = methods.iter().map(|v| match v {
                1 => hinterp::Method::SixteenPoint, 2 => hinterp::Method::FourPoint,
                3 => hinterp::Method::NearestNeighbor, 4 => hinterp::Method::Average4,
                5 => hinterp::Method::Average16, 6 => hinterp::Method::WtAverage4,
                7 => hinterp::Method::WtAverage16,
                8 => hinterp::Method::Search, _ => panic!("unknown method") }).collect();
            for [x,y] in points { array(&mut out,&[hinterp::interp_sequence(x,y,&ctx,&methods,0)]); }
        }
    }
    out.flush().unwrap();
    if let Ok(expected)=std::env::var("CASE7_F1_EXPECTED_RECORD") {
        let expected=f1_payload(&fs::read(expected).unwrap(),"post","words");
        let got=fs::read(std::env::var("CASE7_OUTPUT").unwrap()).unwrap();
        assert_eq!(expected.len(),got.len(),"captured native output shape");
        assert_eq!(expected.len()%4,0);
        for (i,(a,b)) in expected.chunks_exact(4).zip(got.chunks_exact(4)).enumerate() {
            assert_eq!(a,b,"captured native output word {i}");
        }
    }

}

// f1's ORCL0001 wire; native precision, signed zeros and NaN payloads stay bytes.
fn f1_payload(bytes:&[u8],phase:&str,name:&str)->Vec<u8> {
    assert!(bytes.len()>=860);assert_eq!(&bytes[..8],b"ORCL0001");
    let h:Vec<_>=bytes[8..92].chunks_exact(4).map(|b|i32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!((h[13],h[14],h[15],h[16],h[17],h[18],h[19],h[20]),(0,1,0,0,0,0,256,0));
    for (i,value) in ["i1_boundary",phase,name].iter().enumerate() {
        assert_eq!(std::str::from_utf8(&bytes[92+i*256..92+(i+1)*256]).unwrap().trim_end(),*value);
    }
    assert!(h[8..13].iter().all(|&v|v==1));
    let n:usize=h[3..8].iter().map(|&v|usize::try_from(v).unwrap()).product();
    let width=usize::try_from(h[1]).unwrap();assert_eq!(bytes.len(),860+n*width);
    if name=="request" {assert_eq!((h[0],h[1],h[2]),(5,1,0));}
    else {assert_eq!((h[0],h[1],h[2]),(3,4,1));}
    bytes[860..].to_vec()
}
