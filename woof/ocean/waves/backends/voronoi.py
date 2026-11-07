"""Device finite-volume transport for native spherical Voronoi cells.

The public protocol is specified by waves/CONTRACT.md. Python constructs the
static device topology and a launch schedule; CUDA does every field operation.
Original implementation from equations, see voronoi_SCIENCE.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class FaceClass:
    edges: object
    cells: object
    edge_positions: object
    length_bound: float
    diffusion_bound: float


class VoronoiBackend:
    name = "voronoi"
    turning_handled = True

    def __init__(self, grid, bathymetry, config=None):
        import cupy as cp
        self.cp = cp
        cfg = dict(config or {})
        self.radius = float(cfg.get("sphere_radius_m", 6371220.0))
        self.cfl = float(cfg.get("geographic_cfl", 0.4))
        self.gse_seconds = float(cfg.get("gse_spreading_seconds", 3600.0))
        self.ratio = float(cfg.get("spectral", {}).get("ratio", 1.1))
        self.first_hz = float(cfg.get("spectral", {}).get("first_hz", .035))
        if not (math.isfinite(self.radius) and self.radius > 0 and
                math.isfinite(self.cfl) and 0 < self.cfl <= .5 and
                math.isfinite(self.gse_seconds) and self.gse_seconds > 0 and
                math.isfinite(self.ratio) and self.ratio > 1 and
                math.isfinite(self.first_hz) and self.first_hz > 0):
            raise ValueError("invalid Voronoi geometry/transport parameters")
        self.speed_bound = 9.80665 / (2 * math.pi * self.first_hz)
        get = grid.get if isinstance(grid, dict) else lambda k: getattr(grid, k)

        def array(key, dtype):
            a = get(key)
            if not hasattr(a, "__cuda_array_interface__"):
                raise TypeError(f"Voronoi {key} must be a device array")
            return cp.ascontiguousarray(a, dtype=dtype)

        # Mesh arrays are the MPAS native convention: radians, metres, and
        # one-based connectivity with zero for a missing cell/edge.
        lat = array("latCell", cp.float64)
        lon = array("lonCell", cp.float64)
        area = array("areaCell", cp.float32)
        self.full_shape = (int(lat.size),)
        nfull = lat.size
        coe = array("cellsOnEdge", cp.int32) - 1
        eoc = array("edgesOnCell", cp.int32) - 1
        counts = array("nEdgesOnCell", cp.int32)
        length = array("dvEdge", cp.float32)
        distance = array("dcEdge", cp.float32)
        angle = array("angleEdge", cp.float32)
        elat = array("latEdge", cp.float64)
        elon = array("lonEdge", cp.float64)
        ne = int(length.size)
        if (lat.ndim != 1 or lon.shape != lat.shape or area.shape != lat.shape or
                coe.shape != (ne, 2) or eoc.ndim != 2 or eoc.shape[0] != nfull or
                counts.shape != lat.shape or any(a.shape != (ne,) for a in
                (distance, angle, elat, elon))):
            raise ValueError("Voronoi mesh array shapes disagree")
        if not hasattr(bathymetry.depth, "__cuda_array_interface__") or not hasattr(bathymetry.wet, "__cuda_array_interface__"):
            raise TypeError("Voronoi depth and wet mask must be device arrays")
        depth = cp.asarray(bathymetry.depth)
        wet = cp.asarray(bathymetry.wet)
        if depth.shape != self.full_shape or wet.shape != self.full_shape:
            raise ValueError("Voronoi bathymetry must use original full-cell order")
        finite = all(bool(cp.all(cp.isfinite(a))) for a in
                     (lat, lon, area, length, distance, angle, elat, elon, depth, wet))
        if (not finite or bool(cp.any(area <= 0)) or bool(cp.any(length <= 0)) or
                bool(cp.any(distance <= 0)) or bool(cp.any((coe < -1) | (coe >= nfull))) or
                bool(cp.any((eoc < -1) | (eoc >= ne))) or
                bool(cp.any((counts < 1) | (counts > eoc.shape[1]))) or
                bool(cp.any((wet > .5) & (depth <= 0)))):
            raise ValueError("invalid finite-volume metrics, connectivity or wet depth")
        self.wet_to_full = cp.flatnonzero(wet > .5).astype(cp.int32)
        self.ncell = int(self.wet_to_full.size)
        if not self.ncell:
            raise ValueError("Voronoi wave grid has no wet cells")
        self.full_to_wet = cp.full(nfull, -1, cp.int32)
        self.full_to_wet[self.wet_to_full] = cp.arange(self.ncell, dtype=cp.int32)
        self.lon = cp.ascontiguousarray(cp.rad2deg(lon[self.wet_to_full]), dtype=cp.float32)
        self.lat = cp.ascontiguousarray(cp.rad2deg(lat[self.wet_to_full]), dtype=cp.float32)
        self.area = cp.ascontiguousarray(area[self.wet_to_full])
        self.depth = cp.ascontiguousarray(depth[self.wet_to_full], dtype=cp.float32)
        self.depth_gradient_east = cp.empty_like(self.depth)
        self.depth_gradient_north = cp.empty_like(self.depth)
        mapped = cp.where(coe >= 0, self.full_to_wet[cp.maximum(coe, 0)], -1)
        keep = cp.flatnonzero(cp.any(mapped >= 0, axis=1)).astype(cp.int32)
        cells = mapped[keep]
        flip = cells[:, 0] < 0
        self.left = cp.ascontiguousarray(cp.where(flip, cells[:, 1], cells[:, 0]))
        self.right = cp.ascontiguousarray(cp.where(flip, cells[:, 0], cells[:, 1]))
        self.ne = int(keep.size)
        original_to_face = cp.full(ne, -1, cp.int32)
        original_to_face[keep] = cp.arange(self.ne, dtype=cp.int32)
        self.faces = cp.ascontiguousarray(cp.where(eoc[self.wet_to_full] >= 0,
            original_to_face[cp.maximum(eoc[self.wet_to_full], 0)], -1))
        self.counts = cp.ascontiguousarray(counts[self.wet_to_full])
        self.max_edges = int(eoc.shape[1])
        self.length = cp.ascontiguousarray(length[keep])
        self.distance = cp.ascontiguousarray(distance[keep])
        self.normal_east = cp.ascontiguousarray(cp.cos(angle[keep]) * cp.where(flip, -1, 1), dtype=cp.float32)
        self.normal_north = cp.ascontiguousarray(cp.sin(angle[keep]) * cp.where(flip, -1, 1), dtype=cp.float32)
        # A missing original cell is an open model boundary; a present dry cell
        # is an absorbing coast. Do not turn the coastline into incident swell.
        open_face = cp.any(coe[keep] < 0, axis=1)
        self.boundary_cells = cp.unique(self.left[open_face]).astype(cp.int32)
        boundary_map = cp.full(self.ncell, -1, cp.int32)
        boundary_map[self.boundary_cells] = cp.arange(self.boundary_cells.size, dtype=cp.int32)
        self.boundary_slot = cp.ascontiguousarray(cp.where(open_face, boundary_map[self.left], -1))
        self.offset_east = cp.empty((self.ncell, self.max_edges), cp.float32)
        self.offset_north = cp.empty_like(self.offset_east)
        self.neighbor_east = cp.empty_like(self.offset_east)
        self.neighbor_north = cp.empty_like(self.offset_east)
        self.neighbor_angle = cp.empty_like(self.offset_east)
        self.left_edge_angle = cp.empty(self.ne, cp.float32)
        self.left_right_angle = cp.empty(self.ne, cp.float32)
        source = Path(__file__).with_suffix(".cu").read_text(encoding="utf-8")
        self.module = cp.RawModule(code=source, options=("--std=c++17", "--use_fast_math"))
        self.functions = {k: self.module.get_function(k) for k in (
            "vor_geometry", "vor_gradient", "vor_flux", "vor_drain", "vor_update", "vor_save")}
        from woof.certify.kernel_manifest import record_module
        record_module("woof.ocean.waves:voronoi", source=source,
                      options=("--std=c++17", "--use_fast_math"), module=self.module)
        self._launch("vor_geometry", self.ncell*self.max_edges, (
            self.faces, self.counts, self.left, self.right,
            cp.ascontiguousarray(lat[self.wet_to_full]), cp.ascontiguousarray(lon[self.wet_to_full]),
            cp.ascontiguousarray(elat[keep]), cp.ascontiguousarray(elon[keep]),
            self.offset_east, self.offset_north, self.neighbor_east, self.neighbor_north,
            self.neighbor_angle, self.left_edge_angle, self.left_right_angle,
            np.int32(self.ncell), np.int32(self.max_edges), np.float64(self.radius)))
        self.all_cells = cp.arange(self.ncell, dtype=cp.int32)
        self.set_depth(self.depth, cp.cuda.get_current_stream())
        perimeter = cp.sum(cp.where(self.faces >= 0, self.length[cp.maximum(self.faces, 0)], 0), axis=1)
        scale = self.area / perimeter
        levels = cp.floor(cp.log2(scale / cp.min(scale))).astype(cp.int32)
        face_levels = cp.minimum(levels[self.left], levels[cp.maximum(self.right, 0)])
        face_levels = cp.where(self.right >= 0, face_levels, levels[self.left])
        self.classes = []
        for level in range(int(cp.max(face_levels)), -1, -1):
            edges = cp.flatnonzero(face_levels == level).astype(cp.int32)
            if not edges.size:
                continue
            ids = cp.concatenate((self.left[edges], self.right[edges]))
            ids = cp.unique(ids[ids >= 0]).astype(cp.int32)
            positions = cp.full(self.ne, -1, cp.int32)
            positions[edges] = cp.arange(edges.size, dtype=cp.int32)
            valid = self.faces[ids] >= 0
            safe = cp.maximum(self.faces[ids], 0)
            active = valid & (positions[safe] >= 0)
            per = cp.sum(cp.where(active, self.length[safe], 0), axis=1)
            diff = cp.sum(cp.where(active, self.length[safe]/self.distance[safe], 0), axis=1)
            self.classes.append(FaceClass(edges, ids, positions,
                float(cp.min(self.area[ids]/per)), float(cp.min(self.area[ids]/diff))))
        self._buffers = None
        self._shape = None
        self._schedule_cache = {}
        self.calls = 0
        self.stage_launches = 0

    def _launch(self, name, n, args, stream=None):
        self.functions[name](((int(n)+127)//128,), (128,), args, stream=stream)

    def _field(self, a, shape, name):
        cp = self.cp
        if not isinstance(a, cp.ndarray) or a.shape != shape or a.dtype != cp.float32 or not a.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous device float32 {shape}")
        return a

    def set_depth(self, depth, stream):
        self._field(depth, (self.ncell,), "depth")
        with stream:
            if depth is not self.depth:
                self.depth[...] = depth
            self._gradient(self.depth, self.depth_gradient_east, self.depth_gradient_north,
                           self.all_cells, 1, False, stream)

    def _gradient(self, state, gx, gy, cells, bins, limited, stream):
        self._launch("vor_gradient", cells.size*bins, (state, gx, gy, cells,
            self.faces, self.counts, self.left, self.right,
            self.offset_east, self.offset_north, self.neighbor_east, self.neighbor_north,
            self.neighbor_angle, np.int32(cells.size), np.int32(self.max_edges), np.int32(bins),
            np.int32(self._shape[2] if limited else 1), np.int32(limited)), stream)

    def max_dt(self, cg_max):
        if not math.isfinite(cg_max) or cg_max <= 0:
            raise ValueError("cg_max must be finite and positive")
        # The core may take a coarse macro-step. The backend partitions its
        # operator, so the finest wet cell does not control the entire globe.
        self.speed_bound = float(cg_max)
        return self.cfl * max(c.length_bound for c in self.classes) / cg_max

    def _schedule(self, dt, nd):
        key = (dt, nd, self.speed_bound)
        if key in self._schedule_cache:
            return self._schedule_cache[key]
        spread = max((math.sqrt(self.ratio)-1/math.sqrt(self.ratio))**2, (2*math.pi/nd)**2)
        diffusion = self.speed_bound**2 * spread * self.gse_seconds / 12
        result = []
        # Strang (1968), SIAM JNA 5, 506-517, doi:10.1137/0705041.
        # Face-class operators each conserve shared-face flux. Palindromic
        # half/full/half composition is second order for smooth solutions.
        order = [(i, dt*.5) for i in range(len(self.classes)-1)]
        order += [(len(self.classes)-1, dt)]
        order += [(i, dt*.5) for i in range(len(self.classes)-2, -1, -1)]
        for i, interval in order:
            group = self.classes[i]
            bound = self.cfl / (self.speed_bound/group.length_bound + 2*diffusion/group.diffusion_bound)
            pieces = max(1, math.ceil(interval/bound))
            result.extend((i, interval/pieces) for _ in range(pieces))
        self._schedule_cache[key] = tuple(result)
        return self._schedule_cache[key]

    def advect(self, state, group_velocity, dt, stream, incident_boundary):
        cp = self.cp
        if state.ndim != 3 or state.shape[0] != self.ncell:
            raise ValueError("Voronoi action must be [nwet,nf,nd]")
        self._field(state, state.shape, "action")
        nf, nd = state.shape[1:]
        self._field(group_velocity, (self.ncell, nf), "group velocity")
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("geographic dt must be finite and nonnegative (zero warms persistent buffers)")
        if incident_boundary is not None:
            self._field(incident_boundary, (self.boundary_cells.size, nf, nd), "incident action")
        with stream:
            if self._buffers is None:
                self._shape = state.shape
                self._buffers = [cp.empty_like(state), cp.empty_like(state)]
                self._old = cp.empty_like(state)
                self._gx, self._gy = (cp.empty_like(state) for _ in range(2))
                # Once face fluxes are formed, gradients are dead until the
                # next RK stage. Reuse one for the donor budget, saving one
                # spectral volume without altering any numerical operation.
                self._drain = self._gy
                self._flux = cp.empty((max(c.edges.size for c in self.classes), nf, nd), cp.float32)
                self._zero_boundary = cp.zeros((max(1, self.boundary_cells.size), nf, nd), cp.float32)
            elif self._shape != state.shape:
                raise ValueError("spectral dimensions cannot change after Voronoi allocation")
            out = self._buffers[1] if state.data.ptr == self._buffers[0].data.ptr else self._buffers[0]
            out[...] = state
            if dt == 0:
                return out
            boundary = self._zero_boundary if incident_boundary is None else incident_boundary
            bins = nf*nd
            for index, h in self._schedule(float(dt), nd):
                c = self.classes[index]
                self._launch("vor_save", c.cells.size*bins,
                    (out, self._old, c.cells, np.int32(c.cells.size), np.int32(bins)), stream)
                for stage in (0, 1):
                    self._gradient(out, self._gx, self._gy, c.cells, bins, True, stream)
                    self._launch("vor_flux", c.edges.size*bins, (
                        out, self._gx, self._gy, group_velocity, self.depth, boundary,
                        c.edges, self.left, self.right, self.boundary_slot,
                        self.normal_east, self.normal_north, self.length, self.distance,
                        self.faces, self.counts, self.offset_east, self.offset_north,
                        self.left_edge_angle, self.left_right_angle,
                        self._flux, np.int32(c.edges.size), np.int32(self.max_edges),
                        np.int32(nf), np.int32(nd), np.float32(self.gse_seconds), np.float32(self.ratio)), stream)
                    self._launch("vor_drain", c.cells.size*bins, (
                        out, self._flux, c.cells, c.edge_positions, self.faces, self.counts,
                        self.left, self.area, self.left_right_angle, self._drain, np.int32(c.cells.size),
                        np.int32(self.max_edges), np.int32(bins), np.int32(nd), np.float32(h)), stream)
                    self._launch("vor_update", c.cells.size*bins, (
                        out, self._old, self._flux, self._drain, c.cells, c.edge_positions,
                        self.faces, self.counts, self.left, self.right, self.area, self.depth, self.left_right_angle,
                        np.int32(c.cells.size), np.int32(self.max_edges), np.int32(bins), np.int32(nd),
                        np.float32(h), np.int32(stage)), stream)
                self.stage_launches += 9
            self.calls += 1
            return out

    def receipt(self):
        arrays = [v for v in vars(self).values() if isinstance(v, self.cp.ndarray)]
        arrays += self._buffers or []
        arrays += [a for c in self.classes for a in (c.edges, c.cells, c.edge_positions)]
        total = sum(a.nbytes for a in {a.data.ptr:a for a in arrays}.values())
        return dict(backend=self.name, wet_cells=self.ncell, faces=self.ne,
            classes=len(self.classes), boundary_cells=int(self.boundary_cells.size),
            device_bytes=total, device_bytes_per_wet_cell=total/self.ncell,
            calls=self.calls, stage_launches=self.stage_launches,
            transport="limited linear finite volume; SSPRK2; symmetric face-class splitting",
            directional_transport="parallel transport through each shared edge; conservative periodic-bin interpolation",
            turning_handled=True,
            gse_spreading_seconds=self.gse_seconds, coast="absorbing; no incident action")


def create_backend(grid, bathymetry, config=None):
    return VoronoiBackend(grid, bathymetry, config)
