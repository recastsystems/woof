//! Independent reproduction of Flanner/Zender 2006 Table 4.
use static_fields::snow_aging::*;
fn main(){let published=[([0.,150.],[[43.6,11.4],[7.1,6.7],[4.5,6.1],[3.2,5.8]]),([50.,150.],[[27.5,15.3],[47.1,1.7],[21.,1.8],[11.9,1.9]]),([200.,50.],[[370.6,0.9],[5.2,1.9],[2.5,1.9],[1.5,1.9]]),([200.,300.],[[47.2,11.7],[35.,1.8],[15.5,1.9],[8.8,1.9]])];
std::thread::scope(|s|{let j:Vec<_>=published.into_iter().map(|(gd,p)|s.spawn(move||{for(i,tc)in[-50.,-20.,-10.,0.].iter().enumerate(){let v=trajectory(tc+273.15,gd[0],gd[1],200,40);let fit=fit(&v);let coarse=trajectory(tc+273.15,gd[0],gd[1],100,20);println!("T_C={tc} G={} rho={} tau={} k={} paper_tau={} paper_k={} re_14d_um={} bin_resolution_re_relative={}",gd[0],gd[1],fit[0],fit[1],p[i][0],p[i][1],v[336]*1e6,(coarse[336]/v[336]-1.).abs());}})).collect();for j in j {j.join().unwrap();}});
}
