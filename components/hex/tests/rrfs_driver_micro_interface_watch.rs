//! External observation of the pristine RRFS complete microphysics handoff in both directions.
//! MPAS-Model-rrfs/src/core_atmosphere/physics/mpas_atmphys_interface.F:643-1625
//! at 1d6e370e. Ptrace and GNU descriptors reuse r1/r12 observer mechanisms.
//! Only breakpoint instruction bytes change; numerical storage is read-only.
//! Global IDs, full pool contents and hidden callbacks are NOT inferred.
use std::{collections::BTreeMap,env,fs,io::{self,Write},path::Path,process::{Command,exit},os::unix::process::CommandExt,ffi::c_void};
unsafe extern "C" {fn ptrace(r:i32,p:i32,a:*mut c_void,d:*mut c_void)->i64;fn waitpid(p:i32,s:*mut i32,o:i32)->i32;fn __errno_location()->*mut i32;fn process_vm_readv(p:i32,l:*const IoVec,nl:usize,r:*const IoVec,nr:usize,f:usize)->isize;}
#[repr(C)] struct IoVec{base:*mut c_void,len:usize}
#[repr(C)] #[derive(Default)] struct Registers{r15:u64,r14:u64,r13:u64,r12:u64,rbp:u64,rbx:u64,r11:u64,r10:u64,r9:u64,r8:u64,rax:u64,rcx:u64,rdx:u64,rsi:u64,rdi:u64,orig_rax:u64,rip:u64,cs:u64,eflags:u64,rsp:u64,ss:u64,fs_base:u64,gs_base:u64,ds:u64,es:u64,fs:u64,gs:u64}
fn bad(s:&str)->io::Error{io::Error::other(s)}
fn trace(r:i32,p:i32,a:u64,d:u64)->io::Result<i64>{unsafe{*__errno_location()=0;let v=ptrace(r,p,a as *mut _,d as *mut _);if v==-1&&*__errno_location()!=0{Err(io::Error::last_os_error())}else{Ok(v)}}}
fn word(p:i32,a:u64)->io::Result<u64>{Ok(trace(2,p,a,0)? as u64)}
fn wait(p:i32)->io::Result<i32>{let mut s=0;if unsafe{waitpid(p,&mut s,0)}!=p{return Err(io::Error::last_os_error())}Ok(s)}
fn arg(p:i32,r:&Registers,i:usize)->io::Result<u64>{if i<6{Ok([r.rdi,r.rsi,r.rdx,r.rcx,r.r8,r.r9][i])}else{word(p,r.rsp+8*(i as u64-5))}}
fn bytes(p:i32,a:u64,n:usize)->io::Result<Vec<u8>>{if n>256*1024*1024{return Err(bad("capture extent exceeds 256MiB"))}let mut v=vec![0;n];let mut i=0;while i<n{let l=IoVec{base:v[i..].as_mut_ptr() as *mut _,len:n-i};let r=IoVec{base:(a+i as u64) as *mut _,len:n-i};let got=unsafe{process_vm_readv(p,&l,1,&r,1,0)};if got<=0{return Err(io::Error::last_os_error())}i+=got as usize;}Ok(v)}
fn text(p:i32,a:u64,n:usize)->io::Result<String>{if n>512{return Err(bad("native key too long"))}let v=bytes(p,a,n)?;let s=String::from_utf8(v).map_err(|_|bad("invalid native key"))?.trim_end().to_owned();if !s.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_'){return Err(bad(&format!("unsafe native key {s:?}, address={a:x}, length={n}")))}Ok(s)}
#[derive(Clone)] struct Array{at:u64,code:u32,width:usize,shape:Vec<usize>,lower:Vec<i64>,stride:Vec<i64>}
impl Array{
 fn scalar(at:u64,code:u32,width:usize)->Self{Self{at,code,width,shape:vec![],lower:vec![],stride:vec![]}}
 fn descriptor(p:i32,a:u64,rank:usize,code:u32)->io::Result<Option<Self>>{let d=bytes(p,a,40+24*rank)?;let u=|i|u64::from_le_bytes(d[i..i+8].try_into().unwrap());if u(0)==0{return Ok(None)}if u(16)!=4||d[28]!=rank as u8||d[29]!=(if code==1{3}else{1})||u(32)!=4{return Err(bad("GNU descriptor ABI mismatch"))}let mut shape=vec![];let mut lower=vec![];let mut stride=vec![];for i in 0..rank{let lo=u(48+24*i) as i64;let hi=u(56+24*i) as i64;if hi<lo-1||hi-lo>10000000{return Err(bad("native bounds invalid"))}shape.push((hi-lo+1) as usize);lower.push(lo);stride.push((u(40+24*i) as i64)*4);}Ok(Some(Self{at:u(0),code,width:4,shape,lower,stride}))}
 fn read(&self,p:i32)->io::Result<Vec<u8>>{let n=self.shape.iter().try_fold(self.width,|n,&d|n.checked_mul(d)).ok_or_else(||bad("extent overflow"))?;let mut expected=self.width as i64;let mut contiguous=true;for(&s,&st)in self.shape.iter().zip(&self.stride){if s>1&&expected!=st{contiguous=false}expected*=s as i64;}if contiguous{return bytes(p,self.at,n)}let mut v=Vec::with_capacity(n);for i in 0..n/self.width{let mut j=i;let mut off=0i64;for(&s,&st)in self.shape.iter().zip(&self.stride){off+=(j%s)as i64*st;j/=s;}v.extend(bytes(p,(self.at as i128+off as i128).try_into().map_err(|_|bad("strided address overflow"))?,self.width)?);}Ok(v)}
 fn save(&self,p:i32,path:&Path,name:&str,phase:&str,event:i32)->io::Result<()>{
  let payload=self.read(p)?;let mut f=fs::OpenOptions::new().create_new(true).write(true).open(path)?;
  f.write_all(b"ORCL0001")?;let mut h=[0i32;21];h[0]=self.code as i32;h[1]=self.width as i32;h[2]=self.shape.len()as i32;
  h[3..8].fill(1);h[8..13].fill(1);for(i,&v)in self.shape.iter().enumerate(){h[3+i]=i32::try_from(v).map_err(|_|bad("shape exceeds record format"))?;h[8+i]=i32::try_from(self.lower[i]).map_err(|_|bad("lower bound exceeds record format"))?;}h[13]=event;h[19]=256;
  for v in h{f.write_all(&v.to_le_bytes())?;}let file=path.file_name().and_then(|s|s.to_str()).ok_or_else(||bad("record file name"))?;let boundary=file.split(".s").next().ok_or_else(||bad("record boundary"))?;for s in [boundary,phase,name]{if s.len()>256{return Err(bad("record name overflow"))}let mut text=[b' ';256];text[..s.len()].copy_from_slice(s.as_bytes());f.write_all(&text)?;}f.write_all(&payload)
 }
}

#[derive(Clone)]enum Kind{Entry{to:bool},Get{rank:usize,code:u32,mode:u8},Returned{rank:usize,code:u32,mode:u8,name:String,out:u64},Done}
struct Trap{byte:u8,kind:Kind}
fn setbyte(p:i32,a:u64,b:u8)->io::Result<()>{let w=word(p,a)?;trace(5,p,a,(w&!255)|b as u64)?;Ok(())}
fn arm(p:i32,t:&mut BTreeMap<u64,Trap>,a:u64,k:Kind)->io::Result<()>{if t.contains_key(&a){return Err(bad("overlapping observer trap"))}let b=word(p,a)?as u8;setbyte(p,a,204)?;t.insert(a,Trap{byte:b,kind:k});Ok(())}
fn step(p:i32,r:&mut Registers,a:u64,b:u8,rearm:bool)->io::Result<()>{setbyte(p,a,b)?;r.rip=a;trace(13,p,0,r as *mut _ as u64)?;trace(9,p,0,0)?;if wait(p)?!=(5<<8|127){return Err(bad("native single step failed"))}if rearm{setbyte(p,a,204)?}Ok(())}
struct Active{prefix:String,values:BTreeMap<String,Array>,pools:BTreeMap<u64,String>,event:i32,absent:Vec<String>}
fn observe(executable:&Path,output:&Path)->io::Result<()>{
 let executable=fs::canonicalize(executable)?;fs::create_dir_all(output)?;
 let nm=Command::new("nm").arg(&executable).output()?;if !nm.status.success(){return Err(bad("nm failed"))}
 let symbols=String::from_utf8_lossy(&nm.stdout).lines().filter_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()==3{Some((v[2].to_owned(),u64::from_str_radix(v[0],16).ok()?))}else{None}}).collect::<BTreeMap<_,_>>();
 let mut cmd=Command::new(&executable);unsafe{cmd.pre_exec(||{trace(0,0,0,0)?;Ok(())});}let child=cmd.spawn()?;let pid=child.id()as i32;
 if wait(pid)?&255!=127{return Err(bad("missing exec stop"))}trace(0x4200,pid,0,0x100000)?;
 let maps=fs::read_to_string(format!("/proc/{pid}/maps"))?;let bias=maps.lines().find_map(|l|{let v=l.split_whitespace().collect::<Vec<_>>();if v.len()>=6&&v[5]==executable.to_string_lossy()&&v[2]=="00000000"{u64::from_str_radix(v[0].split('-').next()?,16).ok()}else{None}}).ok_or_else(||bad("load bias absent"))?;
 let sym=|n:&str|symbols.get(n).copied().map(|v|v+bias).ok_or_else(||bad(&format!("missing symbol {n}")));
 let from=sym("__mpas_atmphys_interface_MOD_microphysics_from_mpas")?;
 let to=sym("__mpas_atmphys_interface_MOD_microphysics_to_mpas")?;
 let end=|start|symbols.values().map(|v|v+bias).filter(|&v|v>start).min().ok_or_else(||bad("routine upper bound"));
 let ranges=[(from,end(from)?),(to,end(to)?)];
 let mut probes=vec![];for rank in 1..=3{for(code,ty)in[(1,"real"),(3,"int")]{probes.push((sym(&format!("__mpas_pool_routines_MOD_mpas_pool_get_array_{rank}d_{ty}"))?,Kind::Get{rank,code,mode:0}));}}
 probes.push((sym("__mpas_pool_routines_MOD_mpas_pool_get_array_0d_int")?,Kind::Get{rank:0,code:3,mode:0}));
 probes.push((sym("__mpas_pool_routines_MOD_mpas_pool_get_dimension_0d")?,Kind::Get{rank:0,code:3,mode:1}));
 probes.push((sym("__mpas_pool_routines_MOD_mpas_pool_get_config_char")?,Kind::Get{rank:0,code:5,mode:2}));
 probes.push((sym("__mpas_pool_routines_MOD_mpas_pool_get_config_logical")?,Kind::Get{rank:0,code:4,mode:2}));
 let mut traps=BTreeMap::new();arm(pid,&mut traps,from,Kind::Entry{to:false})?;arm(pid,&mut traps,to,Kind::Entry{to:true})?;let mut active:Option<Active>=None;let mut event=0;trace(7,pid,0,0)?;
 loop{let status=wait(pid)?;if status&127==0{if active.is_some(){return Err(bad("native exited inside coupling"))}exit((status>>8)&255)}if status&255!=127{exit(128+(status&127))}let sig=(status>>8)&255;let mut r=Registers::default();trace(12,pid,0,&mut r as *mut _ as u64)?;let at=r.rip-1;
  let Some(trap)=traps.get(&at)else{trace(7,pid,0,sig as u64)?;continue};if sig!=5{trace(7,pid,0,sig as u64)?;continue}let byte=trap.byte;let kind=trap.kind.clone();let mut rearm=true;
  if matches!(kind,Kind::Get{..}){let caller=word(pid,r.rsp)?;if !ranges.iter().any(|&(lo,hi)|caller>=lo&&caller<hi){step(pid,&mut r,at,byte,true)?;trace(7,pid,0,0)?;continue}}
  match kind{
   Kind::Entry{to}=>{if active.is_some(){return Err(bad("nested coupling boundary"))}let mut pools=BTreeMap::new();for(i,n)in[(0,"block.configs"),(1,"mesh"),(2,"state"),(4,"diag"),(5,"diag_physics"),(6,"tend_physics")]{pools.insert(arg(pid,&r,i)?,n.to_owned());}if to{pools.insert(arg(pid,&r,7)?,"tend".to_owned());}
    let boundary=if to{"microphysics_to_mpas"}else{"microphysics_from_mpas"};
    let mut a=Active{prefix:format!("boundary__{boundary}.s0000000000.e{event:010}.r000000"),values:BTreeMap::new(),pools,event,absent:vec![]};event+=1;
    for(i,n)in[(3,"argument.time_lev"),(if to{8}else{7},"argument.its"),(if to{9}else{8},"argument.ite")]{a.values.insert(n.into(),Array::scalar(arg(pid,&r,i)?,3,4));}
    for(n,rank)in [("cldfrac_p",3),("dz_p",3),("evapprod_p",3),("max_hail_diameter_column_p",2),("max_hail_diameter_sfc_p",2),("muc_p",2),("nc_p",3),("nccn_p",3),("ng_p",3),("nh_p",3),("ni_p",3),("nifa2d_p",2),("nifa_p",3),("nr_p",3),("ns_p",3),("ntc_p",2),("nwfa2d_p",2),("nwfa_p",3),("pi_p",3),("pres_p",3),("qc_p",3),("qcbl_p",3),("qg_p",3),("qh_p",3),("qi_p",3),("qr_p",3),("qs_p",3),("qv_p",3),("rainprod_p",3),("recloud_p",3),("refl10cm_p",3),("reice_p",3),("resnow_p",3),("rho_p",3),("th_p",3),("volg_p",3),("volh_p",3),("w_p",3),("z_p",3),("zgw_p",3),("zhw_p",3),("zrw_p",3)]{let at=sym(&format!("__mpas_atmphys_vars_MOD_{n}"))?;if let Some(v)=Array::descriptor(pid,at,rank,1)?{a.values.insert(format!("vars.{n}"),v);}else{a.absent.push(format!("vars.{n}"));}}
    for n in ["kts","kte","jts","jte"]{a.values.insert(format!("vars.{n}"),Array::scalar(sym(&format!("__mpas_atmphys_vars_MOD_{n}"))?,3,4));}
    a.values.insert("vars.dt_dyn".into(),Array::scalar(sym("__mpas_atmphys_vars_MOD_dt_dyn")?,1,4));
    for(n,v)in &a.values{v.save(pid,&output.join(format!("{}.pre.{n}.bin",a.prefix)),n,"pre",a.event)?;}
    arm(pid,&mut traps,word(pid,r.rsp)?,Kind::Done)?;active=Some(a);for(ip,k)in&probes{arm(pid,&mut traps,*ip,k.clone())?;}
   }
   Kind::Get{rank,code,mode}=>{let a=active.as_ref().ok_or_else(||bad("getter outside coupling"))?;let pool=a.pools.get(&arg(pid,&r,0)?).ok_or_else(||bad("unbound native pool address"))?;
    let field=text(pid,arg(pid,&r,1)?,arg(pid,&r,if mode==1{3}else{4})?as usize)?;
    let name=if mode==0{let tl=arg(pid,&r,3)?;format!("{pool}.{field}.t{}",if tl==0{1}else{word(pid,tl)?as i32})}else{format!("{pool}.{field}")};
    arm(pid,&mut traps,word(pid,r.rsp)?,Kind::Returned{rank,code,mode,name,out:arg(pid,&r,2)?})?;
   }
   Kind::Returned{rank,code,mode,name,out}=>{rearm=false;let a=active.as_mut().unwrap();let value=if rank==0{let loc=word(pid,out)?;if loc==0{None}else{Some(Array::scalar(loc,code,if mode==2&&code==5{512}else{4}))}}else{Array::descriptor(pid,out,rank,code)?};
    if let Some(v)=value{if let Some(old)=a.values.get(&name){if old.at!=v.at||old.shape!=v.shape{return Err(bad("native duplicate key has different storage"))}}else{v.save(pid,&output.join(format!("{}.pre.{name}.bin",a.prefix)),&name,"pre",a.event)?;a.values.insert(name,v);}}else{a.absent.push(name);}
   }
   Kind::Done=>{rearm=false;let a=active.take().unwrap();for(n,v)in&a.values{v.save(pid,&output.join(format!("{}.post.{n}.bin",a.prefix)),n,"post",a.event)?;}
    let mut f=fs::File::create(output.join(format!("{}.scope.txt",a.prefix)))?;writeln!(f,"pristine external observation; MPI1; caller entry/return; native_model_step_unknown=true; global_ids_known=false; unqueried_pool_contents_captured=false; callback_state_complete=false; hidden_state_complete=false; all allocated referenced module scratch and direct pool records included; absent allocations explicitly listed; unrelated/private callback storage unknown")?;for n in a.absent{writeln!(f,"absent={n}")?;}
    for(ip,_)in&probes{let t=traps.remove(ip).ok_or_else(||bad("probe trap lost"))?;setbyte(pid,*ip,t.byte)?;}
   }
  }
  step(pid,&mut r,at,byte,rearm)?;if !rearm{traps.remove(&at);}trace(7,pid,0,0)?;
 }
}
fn main()->io::Result<()>{let args=env::args_os().collect::<Vec<_>>();if args.len()!=3{return Err(bad("rrfs-driver-watch NATIVE_EXECUTABLE CAPTURE_DIRECTORY"))}observe(Path::new(&args[1]),Path::new(&args[2]))}
