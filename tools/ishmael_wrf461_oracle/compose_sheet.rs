//! Compose existing Rust-rendered PNGs without changing pixels or map geometry.
use std::{ffi::CString,os::raw::{c_char,c_int,c_void}};
#[repr(C)] struct PngImage {opaque:*mut c_void,version:u32,width:u32,height:u32,format:u32,flags:u32,colormap_entries:u32,warning_or_error:u32,message:[c_char;64]}
#[link(name="png16")] unsafe extern "C" {
 fn png_image_begin_read_from_file(image:*mut PngImage,file:*const c_char)->c_int;
 fn png_image_finish_read(image:*mut PngImage,background:*const c_void,buffer:*mut c_void,stride:c_int,colormap:*mut c_void)->c_int;
 fn png_image_write_to_file(image:*mut PngImage,file:*const c_char,convert:c_int,buffer:*const c_void,stride:c_int,colormap:*const c_void)->c_int;
 fn png_image_free(image:*mut PngImage);
}
fn blank()->PngImage {PngImage{opaque:std::ptr::null_mut(),version:1,width:0,height:0,format:3,flags:0,colormap_entries:0,warning_or_error:0,message:[0;64]}}
fn c(s:&str)->CString {CString::new(s).unwrap()}
fn read(path:&str)->(u32,u32,Vec<u8>) {unsafe{let mut img=blank();assert_ne!(png_image_begin_read_from_file(&mut img,c(path).as_ptr()),0);img.format=3;assert!(u64::from(img.width)*u64::from(img.height)<100_000_000);let mut bytes=vec![0;(img.width*img.height*4) as usize];assert_ne!(png_image_finish_read(&mut img,std::ptr::null(),bytes.as_mut_ptr().cast(),0,std::ptr::null_mut()),0);png_image_free(&mut img);(img.width,img.height,bytes)}}
fn stack(images:&[(u32,u32,Vec<u8>)])->(u32,u32,Vec<u8>){assert!(!images.is_empty());let width=images[0].0;assert!(images.iter().all(|v|v.0==width));let height=images.iter().map(|v|v.1).sum();let pixels=images.iter().flat_map(|v|v.2.iter().copied()).collect();(width,height,pixels)}
fn main(){let a:Vec<_>=std::env::args().collect();assert!(a.len()>=4,"output.png input.png input.png [...]");let images:Vec<_>=a[2..].iter().map(|p|read(p)).collect();let (width,height,bytes)=stack(&images);unsafe{let mut img=blank();img.width=width;img.height=height;assert_ne!(png_image_write_to_file(&mut img,c(&a[1]).as_ptr(),0,bytes.as_ptr().cast(),0,std::ptr::null()),0);png_image_free(&mut img);}println!("{{\"width\":{width},\"height\":{height},\"panels\":{},\"operation\":\"vertical concatenation with unchanged RGBA pixels\"}}",images.len());}
#[cfg(test)] mod tests {use super::stack;#[test]fn keeps_pixels_in_panel_order(){let a=(1,1,vec![1,2,3,255]);let b=(1,1,vec![4,5,6,255]);assert_eq!(stack(&[a,b]),(1,2,vec![1,2,3,255,4,5,6,255]));}}
