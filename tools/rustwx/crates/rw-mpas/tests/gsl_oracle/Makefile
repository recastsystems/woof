# Run with make -j4 REF=<MPAS-Model v8.4.2 checkout>/src/core_init_atmosphere; pristine sources are copied
# into the work directory.
FC = gfortran
CC = gcc
RK ?= 4
REF ?= $(error set REF to an MPAS-Model v8.4.2 checkout's src/core_init_atmosphere)
HARNESS ?= .
STRICT = -O3 -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations $(EXTRA)
FFLAGS = $(STRICT) -ffree-form -ffree-line-length-none -fconvert=big-endian -cpp -DRK=$(RK)
all: oracle
%.F: $(REF)/%.F
	cp $< $@
read_geogrid.c: $(REF)/read_geogrid.c
	cp $< $@
stubs.o: $(HARNESS)/stubs.F
	$(FC) $(FFLAGS) -c $< -o $@
mpas_gsl_oro_data_sm_scale.o: mpas_gsl_oro_data_sm_scale.F stubs.o
	$(FC) $(FFLAGS) -c $< -o $@
mpas_gsl_oro_data_lg_scale.o: mpas_gsl_oro_data_lg_scale.F stubs.o
	$(FC) $(FFLAGS) -c $< -o $@
mpas_init_atm_gwd_gsl.o: mpas_init_atm_gwd_gsl.F mpas_gsl_oro_data_sm_scale.o mpas_gsl_oro_data_lg_scale.o
	$(FC) $(FFLAGS) -c $< -o $@
driver.o: $(HARNESS)/driver.F mpas_init_atm_gwd_gsl.o
	$(FC) $(FFLAGS) -c $< -o $@
read_geogrid.o: read_geogrid.c
	$(CC) $(STRICT) -c $< -o $@
oracle: driver.o mpas_init_atm_gwd_gsl.o mpas_gsl_oro_data_sm_scale.o mpas_gsl_oro_data_lg_scale.o stubs.o read_geogrid.o
	$(FC) $(STRICT) $^ -o $@
