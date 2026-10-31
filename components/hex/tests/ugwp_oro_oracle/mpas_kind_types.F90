! Minimal MPAS framework kinds: mpas_kind_types.F: RKIND single, R8KIND double.
module mpas_kind_types
implicit none
integer,parameter :: RKIND=selected_real_kind(6),R8KIND=selected_real_kind(12)
end module
