# gwd_opt = 1 (KIM orographic drag) WRF v4.8.0 replay fixture

Written 2026-11-09 by `tools/gwdo_kim_wrf480_oracle` (`make_inputs.py` and
`build.sh` on a development machine, then `compare_gpu.py --write-fixture` on a development machine's RTX 4090).

- **Reference words.** The words are the native WRF v4.8.0 strict referee's own
  object files, linked into `run_gwdo_kim480.F90`. The source build is
  `~/wrf480-port/build/WRF-4.8.0-serial-pristine`, commit `06d4240ae989`,
  MMM-physics `550b5b49`.
  - Recipe: GNU Fortran 13.3.0, glibc 2.39, FCOPTIM `-O2 -fno-fast-math
    -ffp-contract=off -fno-tree-vectorize`, container `localhost/wrfref2404:1`.
  - Object and source hashes: `tools/gwdo_kim_wrf480_oracle/OBJECTS.sha256`.
  - `bl_gwdo.o` imports scalar `atan2f`, `expf`, `lroundf` and `powf` only (no
    `_ZGV*`).
- **Sets.**
  - `L50`, `L90`: the v4.6.1 oracle's 149 regime columns
    (`tests/data/oracles/gwdo_kim`, the same words) plus ELVMAX.
  - `L60top3`: the same regimes on 60 levels with a 3 hPa top.
  - Cases for these sets: six (dx, dt) cases × {(gwd_dx_factor 2, gwd_if_nonhyd
    true), (1, false)}.
  - `real-*`: ArWen HRRR-recipe history frames from
    `datasets/era5-hrrr-arwen-pairs/v1`, with the HRRR static's VAR/CON/OA/OL
    and ELVMAX from the 30" terrain. Each set keeps every 97th column plus up
    to 400 of the columns WRF dragged. Cases: 3 km / 20 s at (2, true) and
    (1, false), and 9 km / 54 s at (2, true).
- **Full replay at the time of writing.** It covers all 57,096 columns of each
  real frame.
  - WRF v4.8.0 against `gwdo_version = "wrf_480"`: 139,841,832 words, 0 not
    bit-identical (strict build, `GPUWM_WRF_EXACT=1`, RTX 4090 sm_89,
    NVRTC 13.5).
  - The same columns against the WRF v4.7.1 referee's objects
    (`WRF-4.7.1-serial-pristine`, `f52c197e`) and the frozen `"wrf_471"`
    kernel: 92,987,700 words, 0 not bit-identical.

## Reference output hashes (`build.sh` receipts, full sets)

- `run_gwdo_kim480.F90` `005977597212cf722d9f0ce940c72a9d196e1a63ba5993266fe7da1ba69d320e`
- `run_gwdo_kim480` `9df95d012fc10d68cf7b7881e3b55fee504033ac0f25b1afbae8db0b4c52cae5`
- `run_gwdo_kim471` `2f96477b3c02f6e3c68d197fc2a4765c66f0a7560d42e79da388d400fbd88ca8`
- `gwdo-kim471-wrf471-L50.bin` `da8644318577feda586e1fcd3827add6acfec97eebbb2bfc499cce009700e7f9`
- `gwdo-kim471-wrf471-L60top3.bin` `32cdc742866c83e6ea72eba41ac84b6c4c425b2a8c5d1fc675690f281b928268`
- `gwdo-kim471-wrf471-L90.bin` `9692ce3264cb391c2d57709dd72ffa01298961684fb0e8c854ae2e2f95807fc4`
- `gwdo-kim471-wrf471-real-2023-12-10_18_00_00.bin` `e2af3ac2c48620b32db3941df3f11f459da44c05e139a7d19e66fe09e5c672da`
- `gwdo-kim471-wrf471-real-2023-12-11_06_00_00.bin` `73944b030b1661ecc517a2c503bc8ae5a902515e94f8ff080217b0e0ee44c7f0`
- `gwdo-kim471-wrf471-real-2024-04-25_18_00_00.bin` `a0773cd0ccc11cff6d8c609a2361731cdb69b95fc4be24bc9b65a37a70567273`
- `gwdo-kim471-wrf471-real-2024-04-26_00_00_00.bin` `9ebbee18ede3e685c5ff19da52dec3f70582ab32ba5921a573a06054af7fc952`
- `gwdo-kim480-wrf480-L50.bin` `3dd902e4a237b67b04b84508c8235aca724f9e7c12d032b81abf58b6e7bbed62`
- `gwdo-kim480-wrf480-L60top3.bin` `9f383c681bb5efd923cdd269f9e6dc64186b24db65565fa935ce0a49de8d3286`
- `gwdo-kim480-wrf480-L90.bin` `85847be2a614a4b7ba312f941fd35110d4d5bba14e8808133ae72f118b568ba6`
- `gwdo-kim480-wrf480-real-2023-12-10_18_00_00.bin` `17486387ae41009b1e13b85b1537df50b9c82ea2f9402df6f708de84e7603f6b`
- `gwdo-kim480-wrf480-real-2023-12-11_06_00_00.bin` `a9cc1fb60390dc5f8ab568c4e5134bd5560d323d585384806a626f81710b9cb5`
- `gwdo-kim480-wrf480-real-2024-04-25_18_00_00.bin` `fe078489b25e6f6f68ec9b134dcab5b8b6c80b835a8b74d009566b5208a2965e`
- `gwdo-kim480-wrf480-real-2024-04-26_00_00_00.bin` `82a98348a59af4c276346187e00a3fcea335f6fc37f03c9a1c72b14ae6e747da`
- `gwdo-kim471-inputs-L50.bin` `2c427484aa7f5d12bce834cdb50250f215689ca962817679364dfad46eeb95ef`
- `gwdo-kim471-inputs-L60top3.bin` `622f818deb3d67352f3636ecc15e52b2f2c4f66ed5d83c5efee0dc808bf146ff`
- `gwdo-kim471-inputs-L90.bin` `a76562b0196f5686ba20e9763be79567a9e811c195af71e6f5c0b6f524e986e1`
- `gwdo-kim471-inputs-real-2023-12-10_18_00_00.bin` `0e7439381d5568c3141a2f4fe2df0f89ef644afed78da50f4506168adbc7db06`
- `gwdo-kim471-inputs-real-2023-12-11_06_00_00.bin` `571c880dc294fa5e782735df641140f04b79262a2d7c0b0de9042b8f284d4001`
- `gwdo-kim471-inputs-real-2024-04-25_18_00_00.bin` `8d3ba5ffb03fadf2e367c9174b6eabdda3262bb42c757d55fb1bfc8dce50a292`
- `gwdo-kim471-inputs-real-2024-04-26_00_00_00.bin` `51ab89c73625540e0859cd4f763a378c6ec7e7191020f345df633ec629e01929`
- `gwdo-kim480-inputs-L50.bin` `1a2d76bff5c74d8a2b49fd611f50df20704693077ca80203807e0a2296bad5ea`
- `gwdo-kim480-inputs-L60top3.bin` `1227ab6d79e0e082caf3a16413b6eaaff5a5a22a7cc01b71d5d217ccbc0a8d73`
- `gwdo-kim480-inputs-L90.bin` `cf633f07d5c882cf3c4e9bc80279f2b4770d3e97219946600dc2e73a2b6cc91d`
- `gwdo-kim480-inputs-real-2023-12-10_18_00_00.bin` `4a8e6967fc13f639a18cb5584aef9e8e70db14841c497801c97b6db5a7874476`
- `gwdo-kim480-inputs-real-2023-12-11_06_00_00.bin` `fc3305d8f675698994a2a90b49a7dea81d4c89eccf889efa765cbdb511825eae`
- `gwdo-kim480-inputs-real-2024-04-25_18_00_00.bin` `3a6747d5a96e8d0177a26ae83b1f0eed4677f9f4ebec91acfc1f44c88b389076`
- `gwdo-kim480-inputs-real-2024-04-26_00_00_00.bin` `86aa5be641747f810b06cdf42074f98549fa28d4580d39751e489bfa7601b087`

## Fixture files

- `gwdo-kim480-inputs-L50.npz` `f1aa6980181d40a7dfdf44090a52cc570c69a181f87ff469db4cf631ac2c243e`
- `gwdo-kim480-inputs-L60top3.npz` `de5386e9e77f443b4d5b5f8877e8c1f2d6cd2c9c9f74d1ff4814b29ee822f4e8`
- `gwdo-kim480-inputs-L90.npz` `1575f3376cbde8d01158905dfbe2e243dbd5c9a1bec90f790065506cda467f10`
- `gwdo-kim480-inputs-real-2023-12-10_18_00_00.npz` `55394d11576c3c5ffa44cb8f1d44cbfad8322c88939d0e7604bb72efea0dc0c9`
- `gwdo-kim480-inputs-real-2023-12-11_06_00_00.npz` `73f9bceadbcaf702f8e382086407ff65ca239d17a4f4c144620114a7d99e7a84`
- `gwdo-kim480-inputs-real-2024-04-25_18_00_00.npz` `f292666655106e33b2e28db1911dcb84bc5b2f01ff5ab565b463a9d989c64e25`
- `gwdo-kim480-inputs-real-2024-04-26_00_00_00.npz` `18213e339596a5b74923a16617ab9dfb3bbd870f423d2d51ff22adf94208faf8`
- `gwdo-kim480-wrf480-L50.npz` `3581882e0852a55fd971c11965b285f6640325e90a77757948afbfcc96666387`
- `gwdo-kim480-wrf480-L60top3.npz` `d9a8d0875a87ab81ab1533721348501312f2357ff8f321c6e78a55573d82af59`
- `gwdo-kim480-wrf480-L90.npz` `92df120a8e44e8af5985b16f12d98bd750d4cc9476c776304e47785e28b23846`
- `gwdo-kim480-wrf480-real-2023-12-10_18_00_00.npz` `d9505f4ae66cda267f2ff238248610cb6eb98c18c8e589a474f6f59152a49176`
- `gwdo-kim480-wrf480-real-2023-12-11_06_00_00.npz` `b487f14578e71d0887852f58ca19e55e4d75bbc8c5d7cfb33a649ff0d6ec9aaf`
- `gwdo-kim480-wrf480-real-2024-04-25_18_00_00.npz` `710954f0904c89982e2a1336cf5892aaf0aceda7a299fe2866af838d9315bb6a`
- `gwdo-kim480-wrf480-real-2024-04-26_00_00_00.npz` `dcd0cd4af61478ccaf45613f6f20894a014805e28db68e914d95776d784747c7`
