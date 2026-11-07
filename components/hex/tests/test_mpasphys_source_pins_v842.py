"""Immutable driver citations checked against the pinned native authority."""
import os
from pathlib import Path
import pytest
import check_mpas_citations as c

PINS = {
    # RP7 now cites the native lowest-layer pressure-thickness statement.
    'bl_mynn_subroutines.F90:3102': c.CitationPin('src/core_atmosphere/physics/physics_mmm/bl_mynn_subroutines.F90', 3102, 3102, '07f68e89a4b9357c6b231a5ecc0760bf19cba4f1fc983a942f58a14d42a6eca4', 1),
    'mpas_atmphys_driver_gwdo.F:735-829': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver_gwdo.F', 735, 829, 'f1356b16357871b94f836a8f8d9636d6ee967454df3326e8d768cf319eb0bbbb', 1),
    'mpas_atmphys_driver_gwdo.F:692-834': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver_gwdo.F', 692, 834, 'ef1d4f9ae5a415c44d1dfeaa749c1d281df63e65f52041e9a265681b29d9f68f', 1),
    'mpas_atmphys_driver_seaice.F:495-536': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver_seaice.F', 495, 536, 'b58620c3647a645edb3ef6c5c04dbbb2c7442793f4386be798bf6bd559bbac66', 1),
    'mpas_atm_time_integration.F:2140-2176': c.CitationPin('src/core_atmosphere/dynamics/mpas_atm_time_integration.F', 2140, 2176, '5daab7795eeaeb0787e3327868c57cbbe878a7074560f9afa5e7d80ad83df2c8', 2),
    'mpas_atm_core.F:1036-1040': c.CitationPin('src/core_atmosphere/mpas_atm_core.F', 1036, 1040, '39a22c7e091698b9cc2ab5dd632e2f8c09cae3797aafcb9b73e9ea4064b83004', 1),
    'mpas_atm_core.F:589-596': c.CitationPin('src/core_atmosphere/mpas_atm_core.F', 589, 596, '5d79dc979d01de073a48ed008566e6f5d1f88c9519f0fa41939f91bffd11f074', 2),
    'mpas_atm_time_integration.F:2770-2816': c.CitationPin('src/core_atmosphere/dynamics/mpas_atm_time_integration.F', 2770, 2816, '0907604596dea20231b733260c7d3e122a8ff7b618a931e4cb074e4919ab9dc0', 1),
    'mpas_atmphys_control.F:310-319': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_control.F', 310, 319, '2cc964f4fe03c8a2b7915805bf85f1bc750ca500e0bb7f22379ef738236e8de4', 3),
    'mpas_atmphys_control.F:92-332': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_control.F', 92, 332, '7673e6018e23e86651a5bb3a5cd75f970afa0efecd979412190933ab0b2f3935', 5),
    'mpas_atmphys_driver.F:111-384': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver.F', 111, 384, 'cbac3811b3a6b8b9b9edc1107bd7fb6994d4eda7c00a8d859897a2dde95c4eab', 4),
    'mpas_atmphys_driver.F:179-184': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver.F', 179, 184, 'bab127a090d9be57c3d885549734070c25a87029cb7ec25fea51ab4caa1c6c22', 1),
    'mpas_atmphys_driver.F:179-384': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver.F', 179, 384, '8fee52143b38bfa42e670bc9a5025cde7c8efe5c86d8869abf12ba553916541b', 2),
    'mpas_atmphys_driver.F:209-365': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver.F', 209, 365, 'c2d407252bbfef7aa97d62c39245f7e6a5459563aa399c9f7b6cf8fd018ce014', 2),
    'mpas_atmphys_driver.F:209-384': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver.F', 209, 384, '21779ed2121f25cd22220a98741395e3eb608bb42b65aca12957a2f879fa9bc3', 12),
    'mpas_atmphys_finalize.F:37-67': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_finalize.F', 37, 67, '77501e5449cfb69b6b50b53e0bc979026af9e9577a68ddb510e03534a85da988', 1),
    'mpas_atmphys_init.F:174-396': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_init.F', 174, 396, 'b276e11abf2636389b8f02ed73931c2b5237915a83245fed2e780cec9d8deaca', 2),
    'mpas_atmphys_init.F:83-481': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_init.F', 83, 481, '7d9f0b84175889168019b2e23c8393a077d3993db8ffd3f5a1e1b8cd04904aaf', 4),
    'mpas_atmphys_manager.F:137-375': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 137, 375, '0d0ed0645f14ca9f11c6a217b5488fad8ad05eb14cebef4638fd0c160c50606c', 1),
    'mpas_atmphys_manager.F:137-766': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 137, 766, '10828362ad269f0bd3eb008995c81ab3666e4a5be3668c4982d23021eb89ff56', 2),
    'mpas_atmphys_manager.F:204-213': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 204, 213, '306e5df22d903d841cee7eadbdf4b54428b8427e1bd3a5f2ae65b2678343a601', 4),
    'mpas_atmphys_manager.F:263-363': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 263, 363, '28f42328cf8dba12836a8717c14d2167f16e01db7b4e459eace1e9a60864fe16', 1),
    'mpas_atmphys_manager.F:263-375': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 263, 375, '7686640c6e36a6b21dab06b591bb02842295cd9e0e6d3bee3b18819579e3abb3', 1),
    'mpas_atmphys_manager.F:385-766': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 385, 766, 'b183febd987d7c8b120314f0716a8059235b6eecfb781b68eeaeadb708812ef3', 1),
    'mpas_atmphys_manager.F:549-644': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_manager.F', 549, 644, '6a434310749cbdfc8f5f82bc89b3cc805c6a44d05990ddfdd3908b401049dbf8', 1),
    'mpas_atmphys_todynamics.F:98-498': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_todynamics.F', 98, 498, 'd8e8bfe14418b9e83ffa8be0642d12aa6f799ce16ee11114fbc541a5086b6293', 1),
    'mpas_atmphys_vars.F:24-953': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_vars.F', 24, 953, 'fdadc20236f2c9d1040b13990eb3e98ed70a9b90bafaa4b44ba2b5e8ab293f6f', 4),
    'mpas_timekeeping.F:1347-1591': c.CitationPin('src/framework/mpas_timekeeping.F', 1347, 1591, 'f1b24ad2adf8527720d7a002a1f84a9ae9a5c4f4886b27940bebd8934335ff26', 1),
    'mpas_timekeeping.F:739-766': c.CitationPin('src/framework/mpas_timekeeping.F', 739, 766, '011cc59e0e08eef962819e4ee40b6bc75cc535d5d57047473b711e49dcee8a5c', 1),
    'mpas_timekeeping.F:876-921': c.CitationPin('src/framework/mpas_timekeeping.F', 876, 921, 'cc93050ee1797a9f67ef6a9cf731bcdce1e5920f28ad4da04a9028cb1a8222cb', 1),
    'mpas_timekeeping.F:925-985': c.CitationPin('src/framework/mpas_timekeeping.F', 925, 985, '552d82b052866107a0cea5abd24107e9929aa0f3b48c50fecd0e698a19a87d06', 3),
    # Correct mode (mpas-rad-pbl-lsm-seaice b655f2e0, pinned on merge/correct-mode 2026-10-14): the RP1-RP6 citations.
    'Registry.xml:2806-2812': c.CitationPin('src/core_atmosphere/Registry.xml', 2806, 2812, 'b5f54c4edbd61683c555ba291a8c759efd51ad40d8f39d467824e5c59fc76573', 1),
    'bl_mynn.F90:766-770': c.CitationPin('src/core_atmosphere/physics/physics_mmm/bl_mynn.F90', 766, 770, '2fb0a59d17d17824645bba3da63e8f34fd80c3b1d3aadfeb61fdf5d2144f8558', 1),
    'mpas_atmphys_driver_gwdo.F:540-547': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver_gwdo.F', 540, 547, 'b7c3438433e5f04f0e7e08a18a05b7c1adb210cf7399e22a662bddc7e8fbcf75', 1),
    'mpas_atmphys_driver_lsm.F:726': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver_lsm.F', 726, 726, '4b2986e4b5f0a3cf7aad77995f3b3da435b29e52718eed4328e497324249e86c', 1),
    'mpas_atmphys_driver_sfclayer.F:428': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_driver_sfclayer.F', 428, 428, 'fa4c64304b6326dc24804e6aba632ba487fe294e6ed12fc579b765e70e8a9434', 1),
    'mpas_atmphys_lsm_noahmpinit.F:289': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_lsm_noahmpinit.F', 289, 289, 'b87c4bd5db28bb1676ea86c743633be8c023c6b862b3d37e78750773cfb4eb46', 1),
    'mpas_atmphys_lsm_shared.F:42-52': c.CitationPin('src/core_atmosphere/physics/mpas_atmphys_lsm_shared.F', 42, 52, 'aa111132b5ff00e18b5bcf8498c4ca779d57dce39b6e0bfa72ae0dbfa8de1cde', 1),
    # Correct mode WB5 (merge/correct-mode-2): the WSM6 rain-out statement.
    'mp_wsm6.F90:1984': c.CitationPin('src/core_atmosphere/physics/physics_mmm/mp_wsm6.F90', 1984, 1984, '77e810fe4814dce0d4aee2cdd58ad2901a040e871c830af4308749cd75ac4d0b', 1),
}

def test_driver_source_pins():
    root=Path(__file__).resolve().parents[1]/'../../woof/hex/mpasphys'
    native=os.environ.get('MPAS_REF')
    if native is None:
        pytest.skip('set MPAS_REF to the pinned MPAS-A v8.4.2 source')
    found={k:tuple(o for o in v if o.python_path.endswith('/driver_v842.py'))
           for k,v in c.citations(root).items()}
    found={k:v for k,v in found.items() if v}
    assert c.check_pins(found,pins=PINS,vendor_root=Path(native)) == []
