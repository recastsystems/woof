//! Complete fork static operator replay probe, RRFSx/MPAS-Model
//! src/core_init_atmosphere/mpas_atm_advection.F:21-939 at 1d6e370e.
use rw_mpas::rrfs_static_operators::{build_operator_fields,OperatorMesh};
use std::io::Write;
#[test]
#[ignore="standalone fork oracle diagnostic requires an owned grid and output path"]
fn operator_probe() {
    let f=netcrust::File::open(std::env::var("RRFS_GRID").unwrap()).unwrap();
    let dims=|n:&str|f.dimension(n).unwrap().len();
    let nc=dims("nCells");let ne=dims("nEdges");let me=dims("maxEdges");
    let radius=6371229f32;
    let rv=|n:&str,scale:bool|->Vec<f64>{f.read_array_f64(n).unwrap().into_values().into_iter().map(|v|{let x=v as f32;f64::from(if scale{x*radius}else{x})}).collect()};
    let iv=|n:&str|->Vec<usize>{f.read_array::<i32>(n).unwrap().into_iter().map(|v|if v>0{v as usize-1}else{usize::MAX}).collect()};
    let xc=rv("xCell",true);let yc=rv("yCell",true);let zc=rv("zCell",true);
    let xv=rv("xVertex",true);let yv=rv("yVertex",true);let zv=rv("zVertex",true);
    let angle=rv("angleEdge",false);let dc=rv("dcEdge",true);
    let counts=f.read_array::<i32>("nEdgesOnCell").unwrap().into_iter().map(|v|v as usize).collect::<Vec<_>>();
    let coc=iv("cellsOnCell");let eoc=iv("edgesOnCell");let voc=iv("verticesOnCell");
    let pairs=|n:&str|->Vec<[usize;2]>{iv(n).chunks_exact(2).map(|v|[v[0],v[1]]).collect()};
    let coe=pairs("cellsOnEdge");let voe=pairs("verticesOnEdge");
    let m=OperatorMesh{n_cells:nc,n_edges:ne,max_edges:me,sphere_radius:radius as f64,n_edges_on_cell:&counts,cells_on_cell:&coc,edges_on_cell:&eoc,cells_on_edge:&coe,vertices_on_cell:&voc,vertices_on_edge:&voe,x_cell:&xc,y_cell:&yc,z_cell:&zc,x_vertex:&xv,y_vertex:&yv,z_vertex:&zv,angle_edge:&angle,dc_edge:&dc};
    let out=build_operator_fields(&m).unwrap();let mut w=std::fs::File::create(std::env::var("RRFS_OUTPUT").unwrap()).unwrap();
    for values in [&out.cell_gradient_coef_x,&out.cell_gradient_coef_y,&out.deriv_two,&out.defc_a,&out.defc_b] {for v in values {w.write_all(&v.to_le_bytes()).unwrap();}}
}
