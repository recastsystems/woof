# Goddard GCE source and licence

The implementation derives from WRF's public source. No competitor source
was used. The WRF public-domain notice allows use for any purpose and asks
that its notice and disclaimer remain in redistributed copies. A copy is
included as [WRF-NOTICE.txt](WRF-NOTICE.txt).

The source pins are WRF v4.6.1, commit
`d66e442fccc04111067e29274c9f9eaccc3cef28`, and WRF v4.7.1, commit
`f52c197ed39d12e087d02c50f412d90d418f6186`.

| File | Upstream source | SHA256 |
| --- | --- | --- |
| GCE, both pins | [4.7.1 module_mp_gsfcgce.F](https://github.com/wrf-model/WRF/blob/v4.7.1/phys/module_mp_gsfcgce.F) | `43e841f78e97a35e261f7d91d83c6ad54c7bf30a74625e933603d2e8d6126ef1` |
| Shared radar, 4.6.1 | [module_mp_radar.F](https://github.com/wrf-model/WRF/blob/v4.6.1/phys/module_mp_radar.F) | `aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688` |
| Shared radar, 4.7.1 | [module_mp_radar.F](https://github.com/wrf-model/WRF/blob/v4.7.1/phys/module_mp_radar.F) | `ca069a12cc149313456fa8ee08a827e0f542768b66ff6c1ece95a40ab208fb88` |
| Licence, both pins | [LICENSE.txt](https://github.com/wrf-model/WRF/blob/v4.7.1/LICENSE.txt) | `b30611d35ad1922bc741e86d21e1a897234c1a9c9e2af28076e021f565dfbec6` |
| Registry, 4.6.1 | [Registry.EM_COMMON](https://github.com/wrf-model/WRF/blob/v4.6.1/Registry/Registry.EM_COMMON) | `28cf7a2d369e898217d7231f4ed19310af7f0bec920f3cb57831e3dc4c80beec` |
| Registry, 4.7.1 | [Registry.EM_COMMON](https://github.com/wrf-model/WRF/blob/v4.7.1/Registry/Registry.EM_COMMON) | `6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a` |

The entire GCE file is byte-identical between these pins. Its `aa1` and
`aa2` coefficient tables are embedded in that file at lines 53-66. It
reads no external coefficient table. The two physics source files contain
no separate restrictive licence notice. No source or table was held for
licensing reasons.

The 4.7.1 radar module adds hail parameters and their initialization.
GCE continues to set and use the existing graupel distribution parameters.
The GCE source identity alone does not establish current GPU equivalence:
the shared radar and the engine's math headers must also be replayed.

The retained oracle hash receipt and README describe earlier synthetic
measurements and three explicitly declared repairs to undefined or
missing native branches. Current real-input measurements across all eight controls
and their scope are recorded separately in [REAL-INPUT.md](REAL-INPUT.md).
