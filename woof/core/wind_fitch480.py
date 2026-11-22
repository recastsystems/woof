"""WRF v4.8.0 Fitch wind farm (phys/module_wind_fitch.F at 06d4240ae989).

Both branches of the 4.8.0 module on InductionFarm's native-table resident
layout; only the column kernel differs (woof/core/kernels/wind_fitch_480.cu):

* windfarm_induction=1 (4.8.0's Registry default): fitch_induction_480;
* windfarm_induction=0: fitch_original_480, the original Fitch drag.

Both write THRUST, which 4.8.0 adds to every Fitch run (Registry package
fitchscheme: state power,thrust). The wrf_471 pin keeps WindFarm and
wind_mav.cu's fitch_induction untouched, so its output is today's.
"""
from woof.core.kernels import get_kernel
from woof.core.wind_induction import InductionFarm


class Fitch480Farm(InductionFarm):
    def __init__(self, cfg, shape, *, induction, **kwargs):
        super().__init__(cfg, shape, **kwargs)
        self.induction = int(induction)
        self.native_kernel = get_kernel(
            "wind_fitch_480", "fitch_induction_480" if self.induction else "fitch_original_480")

    def receipt(self):
        return dict(super().receipt(), induction=self.induction, windfarm_fitch_version="wrf_480")
