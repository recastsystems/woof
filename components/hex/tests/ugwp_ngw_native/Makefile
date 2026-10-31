FC = gfortran
FFLAGS = -O3 -fPIC -ffree-form -ffree-line-length-none -cpp -ffp-contract=off -fno-tree-vectorize -fno-unsafe-math-optimizations --coverage
SOURCES = stubs.F90 cires_ugwpv1_initialize.F cires_tauamf_data.F cires_ugwpv1_module.F cires_ugwpv1_triggers.F cires_ugwpv1_solv2.F bl_ugwpv1_ngw.F wrapper.F90 wrapper_modes.F90 writeback.F90
all: liboracle.so
liboracle.so: $(SOURCES)
	$(FC) $(FFLAGS) -shared $(SOURCES) -o $@
