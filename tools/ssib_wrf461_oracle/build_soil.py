"""Build WRF's unchanged soil routines at the SSiB initialization depths."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
source,work=map(Path,sys.argv[1:])
commit='d66e442fccc04111067e29274c9f9eaccc3cef28'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==commit
subprocess.run(['git','-C',str(source),'diff','--quiet','HEAD','--','share/module_soil_pre.F'],check=True)
raw=(source/'share/module_soil_pre.F').read_bytes(); text=raw.decode()
def routine(kind,name):
    return re.search(r'^\s*'+kind+r' '+name+r'\b.*?^\s*END '+kind+r' '+name+r'\b[^\n]*',text,re.M|re.S|re.I).group(0)
work.mkdir(parents=True,exist_ok=True)
module='module soil_oracle\nimplicit none\ninteger :: em_width=0\nlogical :: hold_ups=.true.\nlogical,external :: skip_middle_points_t\ncharacter(len=32)::current_date="",start_date=""\ncontains\n'
module+=routine('SUBROUTINE','init_soil_depth_8')+'\n'+routine('SUBROUTINE','init_soil_2_real')+'\nend module\n'
module+=routine('FUNCTION','skip_middle_points_t')+'\n'
module+='subroutine wrf_message(message)\ncharacter(len=*)::message\nend subroutine\nsubroutine wrf_error_fatal(message)\ncharacter(len=*)::message\nprint *,message\nstop 1\nend subroutine\n'
(work/'module.f90').write_text(module)
command=['gfortran','-O0','-ffree-form','-ffree-line-length-none','module.f90',str(Path(__file__).with_name('soil_driver.F90').resolve()),'-o','soil_oracle']
subprocess.run(command,cwd=work,check=True)
subprocess.run([str((work/'soil_oracle').resolve())],cwd=work,check=True)
(work/'receipt.json').write_text(json.dumps({'wrf_commit':commit,'source_sha256':hashlib.sha256(raw).hexdigest(),'command':command,'fixture_sha256':hashlib.sha256((work/'soil-columns.bin').read_bytes()).hexdigest(),'routines':['init_soil_depth_8','init_soil_2_real','skip_middle_points_t'],'stubs':['wrf_message','wrf_error_fatal','current_date','start_date'],'columns':64},indent=2)+'\n')
