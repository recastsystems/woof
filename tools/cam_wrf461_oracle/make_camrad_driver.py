"""Append a full WRF camrad column harness to the independently built oracle."""
from pathlib import Path
import sys,re
from make_driver import arguments
src=Path(sys.argv[1]).read_text();out=Path(sys.argv[2]);levels=int(sys.argv[3]) if len(sys.argv)>3 else 32;base=out.read_text();args,decl=arguments(src,'camrad')
params={'ids':1,'ide':3,'jds':1,'jde':2,'kds':1,'kde':33,'ims':1,'ime':2,'jms':1,'jme':1,'kms':1,'kme':33,'its':1,'ite':2,'jts':1,'jte':1,'kts':1,'kte':32,'levsiz':59,'num_months':13,'paerlev':29,'naer_c':13,'cam_abs_dim1':4,'cam_abs_dim2':33,'n_cldadv':3}
text=['subroutine run_camrad(icase)','use module_ra_cam, only: camrad','use module_ra_clwrf_support, only: read_CAMgases','implicit none','integer,intent(in)::icase','integer::i,k,m,group','real(8)::frac,cloud,gas(5)','logical::read_file','integer,parameter::'+','.join(k+'='+str(v) for k,v in params.items())]
for a in args:
 if a in params:continue
 text.append(re.sub(r',?\s*(?:intent\(\w+\)|optional)','',decl[a]))
text.extend(['group=mod(icase-1,8)'])
for a in args:
 if a in params:continue
 text.append(a+'='+('.true.' if decl[a].startswith('logical') else '0'))
text.extend(['dolw=.true.','dosw=.true.','doabsems=.true.','ghg_input=1','yr=2026','julday=15+5*(icase-1)','julian=real(julday-1)+0.25','dt=60.','radt=30.','gmt=6.','degrad=acos(-1.)/180.','solcon=1367.','sf_surface_physics=2',
 'xlat=35.','if(group==5) xlat=80.','if(group==6) xlat=0.','xlong=0.','coszen=0.7','if(group==4) coszen=-0.2','if(group==5) coszen=0.01','if(group==7) coszen=1.e-8','albedo=0.2','emiss=0.98','tsk=300.','xland=1.','alswvisdir=0.15','alswvisdif=0.2','alswnirdir=0.25','alswnirdif=0.3',
 'do k=1,kme','frac=real(k-1,8)/32.','p8w(:,k,1)=exp(log(100000._8)+frac*log((1000._8-0.5_8*(icase-1))/100000._8))','z(:,k,1)=1000.*(k-1)','end do','do k=1,kte','frac=real(kte-k,8)/(kte-1)','p_phy(:,k,1)=0.5*(p8w(:,k,1)+p8w(:,k+1,1))','t_phy(:,k,1)=220._8+75._8*frac**2','if(group==5)t_phy(:,k,1)=190._8+65._8*frac**2','if(group==6)t_phy(:,k,1)=230._8+77._8*frac**2','qv3d(:,k,1)=1.e-6_8+0.014_8*frac**5','pi_phy(:,k,1)=(p_phy(:,k,1)/100000.)**(287./1004.5)','rho_phy(:,k,1)=p_phy(:,k,1)/(287.*t_phy(:,k,1))','dz8w(:,k,1)=1000.','cloud=0.','if(group==1.and.k<=10)cloud=1.','if(group==2.and.((k>=8.and.k<=12).or.(k>=22.and.k<=26)))cloud=0.4_8+0.1_8*mod(k,3)','if(group==7.and.k==1)cloud=1.e-12_8','cldfra(:,k,1)=cloud','qc3d(:,k,1)=1.e-4_8*cloud','qi3d(:,k,1)=2.e-5_8*cloud','end do',
 "open(36,file='ozone_plev.formatted',status='old')",'do k=1,levsiz','read(36,*)pin0(k)','pin0(k)=pin0(k)*100.','do m=1,num_months','ozmixm(:,k,1,m)=2.e-8_8+8.e-6_8*exp(-((log(real(pin0(k),8))-log(10000._8))/1.0_8)**2)','end do','end do','close(36)','m_psp=100000.','m_psn=100000.','do k=1,paerlev','m_hybi0(k)=real(k-1)/(paerlev-1)','aerosolcp(:,k,1,1)=1.e-7*(1.-m_hybi0(k))','if(group==3)aerosolcp(:,k,1,1)=1.e-4*(1.-m_hybi0(k))','aerosolcn(:,k,1,1)=aerosolcp(:,k,1,1)','end do','read_file=.false.',"call read_CAMgases(yr,julian,read_file,'CAM',gas(1),gas(2),gas(3),gas(4),gas(5))", "call dump('camrad',icase,'input:gases',gas)"])
for a in args:
 if a in params:continue
 role='output' if 'intent(out)' in decl[a] else 'input'
 if role=='input':text.append("call dump('camrad',icase,'input:"+a+"',real("+('merge(1,0,' if decl[a].startswith('logical') else '')+"reshape(["+a+"],[size(["+a+"])])"+(')' if decl[a].startswith('logical') else '')+",8))")
text.extend(['call camrad(&',', &\n'.join(','.join(args[i:i+6]) for i in range(0,len(args),6))+')'])
for a in args:
 if a in params:continue
 if 'intent(inout)' in decl[a] or 'intent(out)' in decl[a]:text.append("call dump('camrad',icase,'output:"+a+"',real("+('merge(1,0,' if decl[a].startswith('logical') else '')+"reshape(["+a+"],[size(["+a+"])])"+(')' if decl[a].startswith('logical') else '')+",8))")
text.append('end subroutine run_camrad')
runner='\n'.join(text)
if levels!=32:
 runner=re.sub(r'\b33\b',str(levels+1),runner)
 runner=re.sub(r'\b32\b',str(levels),runner)
base=base.replace('call run_sw(icase)','call run_sw(icase)\ncall run_camrad(icase)').replace('end program',runner+'\nend program')
base=base.replace('subroutine init()\nuse module_ra_cam_support','subroutine init()\nuse module_ra_clwrf_support, only: read_CAMgases\nuse module_ra_cam_support')
base=base.replace('implicit none\ntmelt=',"implicit none\nlogical :: read_file\nreal(8)::gas(5)\nixcldliq=2\nixcldice=3\nozncyc=.true.\nindirect=.true.\nr_universal=6.02214e26*5.67e-8\nread_file=.true.\ncall read_CAMgases(2026,0.,read_file,'CAM',gas(1),gas(2),gas(3),gas(4),gas(5))\ntmelt=")
out.write_text(base)
