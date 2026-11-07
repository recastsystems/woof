from pathlib import Path
import hashlib, json
import argparse
ap=argparse.ArgumentParser(description='Instrument an audited WRF CU_SAS source copy for first-call capture')
ap.add_argument('source',type=Path)
ap.add_argument('output',type=Path)
args=ap.parse_args()
source=args.source
s=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()=='a59026ff2061293ce657e7d56fdcbe2381a615721e2801616997cd3f96456158'
s=s.replace('      DATA IGPVS/0/', '''      ! Lane-only capture. Array sections are written without arithmetic.
      LOGICAL, SAVE :: sas_capture_done = .false.
      LOGICAL :: sas_capture_this
      INTEGER :: sas_ci(256), sas_cj(256), sas_nc, sas_idx, sas_unit
      INTEGER :: sas_di, sas_dj, sas_env_status
      CHARACTER(1024) :: sas_prefix
      CHARACTER(32) :: sas_rank
      DATA IGPVS/0/''',1)
anchor='      DO J=JTS,JTE\n'
insert='''      sas_capture_this = .false.
      call get_environment_variable('WOOF_SAS_CAPTURE', sas_prefix, status=sas_env_status)
      if (sas_env_status == 0 .and. len_trim(sas_prefix) > 0 .and. .not. sas_capture_done) then
        sas_capture_this = .true.
        call get_environment_variable('OMPI_COMM_WORLD_RANK', sas_rank, status=sas_env_status)
        if (sas_env_status /= 0) sas_rank = '0'
        sas_nc = 0
        sas_di = max(1, (ite-its+1)/16)
        sas_dj = max(1, (jte-jts+1)/16)
        do j=jts,jte,sas_dj
          do i=its,ite,sas_di
            if (sas_nc < 256) then
              sas_nc = sas_nc+1
              sas_ci(sas_nc)=i
              sas_cj(sas_nc)=j
            endif
          enddo
        enddo
        open(newunit=sas_unit,file=trim(sas_prefix)//'-'//trim(sas_rank)//'-cases.bin', &
             access='stream',form='unformatted',status='replace')
        write(sas_unit) sas_nc
        do sas_idx=1,sas_nc
          i=sas_ci(sas_idx)
          j=sas_cj(sas_idx)
          write(sas_unit) kte-kts+1,stepcu,dt,real(pgcon_use),xland(i,j)
          write(sas_unit) t3d(i,kts:kte,j),qv3d(i,kts:kte,j),qc3d(i,kts:kte,j),qi3d(i,kts:kte,j)
          write(sas_unit) u3d(i,kts:kte,j),v3d(i,kts:kte,j),w(i,kts:kte+1,j),pcps(i,kts:kte,j)
          write(sas_unit) pi3d(i,kts:kte,j),rho3d(i,kts:kte,j),dz8w(i,kts:kte,j),p8w(i,kts:kte+1,j)
          write(sas_unit) 0
        enddo
        close(sas_unit)
      endif

'''
assert s.count(anchor)>=1;s=s.replace(anchor,insert+anchor,1)
anchor='   END SUBROUTINE CU_SAS'
insert='''      if (sas_capture_this) then
        open(newunit=sas_unit,file=trim(sas_prefix)//'-'//trim(sas_rank)//'-stage3.bin', &
             access='stream',form='unformatted',status='replace')
        write(sas_unit) sas_nc
        do sas_idx=1,sas_nc
          i=sas_ci(sas_idx)
          j=sas_cj(sas_idx)
          write(sas_unit) kte-kts+1
          write(sas_unit) rthcuten(i,kts:kte,j),rqvcuten(i,kts:kte,j),rqccuten(i,kts:kte,j),rqicuten(i,kts:kte,j)
          write(sas_unit) raincv(i,j),pratec(i,j),hbot(i,j),htop(i,j)
        enddo
        close(sas_unit)
        sas_capture_done = .true.
      endif
'''
assert s.count(anchor)==1;s=s.replace(anchor,insert+anchor)
out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'module_cu_sas.F').write_text(s)
(out/'instrumentation.json').write_text(json.dumps({'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'instrumented_sha256':hashlib.sha256(s.encode()).hexdigest(),'profiles_per_rank':256,'arithmetic_changed':False,'word_bytes':4,'byte_order':'native compiler -fconvert value; canonicalize explicitly','capture':'CU_SAS entry inputs and exit outputs, first call only; no array arithmetic'},indent=2)+'\n')
