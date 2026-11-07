use std::{env,fs,io::Write};
fn npy(path:&str,v:&[f32])->Result<(),Box<dyn std::error::Error>> {
 let mut f=fs::File::create(path)?;
 let header=format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",v.len());
 let pad=(64-(10+header.len()+1)%64)%64;
 let h=header+&" ".repeat(pad)+"\n";
 f.write_all(b"\x93NUMPY\x01\x00")?;f.write_all(&(h.len() as u16).to_le_bytes())?;f.write_all(h.as_bytes())?;
 for x in v {f.write_all(&x.to_bits().to_le_bytes())?;} Ok(())
}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let a:Vec<String>=env::args().collect();if a.len()<4{return Err("usage: table_decode text OUTPUT INPUT1 INPUT2 INPUT3 | binary OUTPUT INPUT LEVELS".into())}
 let mut values=Vec::new();
 if a[1]=="numbers"{
  if a.len()!=4{return Err("numbers mode requires one coefficient text file".into())}
  for token in fs::read_to_string(&a[3])?.split(|c:char| c.is_whitespace()||c==',').filter(|s|!s.is_empty()){
   let x:f32=token.parse()?;if !x.is_finite(){return Err("nonfinite coefficient".into())}values.push(x);
  }
 }else if a[1]=="text"{
  if a.len()!=6{return Err("three transmission files are required".into())}
  for path in &a[3..]{
   let src=fs::read_to_string(path)?;let begin=values.len();
   for (i,line) in src.lines().enumerate(){
    let x:f32=line.trim().parse().map_err(|_|format!("{}:{}: invalid transmission number",path,i+1))?;
    if !x.is_finite(){return Err(format!("{}:{}: nonfinite transmission",path,i+1).into())} values.push(x);
   }
   if values.len()-begin!=109*109*3{return Err(format!("{} has wrong transmission shape",path).into())}
  }
 }else if a[1]=="binary"{
  if a.len()!=5{return Err("binary mode requires the run's layer count".into())}
  let n:usize=a[4].parse()?;if n==0||n>81{return Err("layer count must be within 1..81".into())}
  let src=fs::read(&a[3])?;if src.len()<8{return Err("truncated CO2 table".into())}
  let be=u32::from_be_bytes(src[0..4].try_into()?) as usize==(n+1)*4;
  let word=|b:&[u8]|->u32{if be{u32::from_be_bytes(b.try_into().unwrap())}else{u32::from_le_bytes(b.try_into().unwrap())}};
  let mut expected=vec![n+1;2];expected.extend(vec![n;6]);expected.extend(vec![(n+1)*(n+1);6]);expected.extend(vec![n+1;12]);
  let mut pos=0;
  for (rec,len) in expected.into_iter().enumerate(){
   if pos+8+len*4>src.len(){return Err(format!("CO2 record {} is truncated or belongs to another vertical grid",rec).into())}
   let bytes=word(&src[pos..pos+4]) as usize;
   if bytes!=len*4||word(&src[pos+4+bytes..pos+8+bytes]) as usize!=bytes{return Err(format!("CO2 record {} shape/marker mismatch for {} layers",rec,n).into())}
   for b in src[pos+4..pos+4+bytes].chunks_exact(4){let x=f32::from_bits(word(b));if !x.is_finite(){return Err("nonfinite CO2 value".into())}values.push(x);}
   pos+=bytes+8;
  }
  if pos!=src.len(){return Err("CO2 file has unexpected trailing records".into())}
 }else{return Err("unknown table format".into())}
 npy(&a[2],&values)?;println!("{{\"values\":{},\"bytes\":{}}}",values.len(),values.len()*4);Ok(())
}
