"""ctypes bindings for libarcsim (batched physics) and libarcrender (SDL/OpenGL).

The shared libraries are built by `make` next to this file. All arrays crossing
the boundary are contiguous float64 / int32 / uint32 numpy arrays.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent

_f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_u32 = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
_vp = ctypes.c_void_p

ROOT_SIZE = 13  # pos(3) quat wxyz(4) linvel(3) angvel(3)


def _load(name: str) -> ctypes.CDLL:
    path = _HERE / name
    if not path.exists():
        raise OSError(f"{path} not found: build the native libraries with `make`")
    return ctypes.CDLL(str(path))


def _declare(lib: ctypes.CDLL, name: str, restype, *argtypes) -> None:
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)


@functools.cache
def sim_lib() -> ctypes.CDLL:
    lib = _load("libarcsim.so")
    _declare(lib, "arc_sim_create", _vp, ctypes.c_int, ctypes.c_double, ctypes.c_int,
             ctypes.c_int, ctypes.c_double)
    _declare(lib, "arc_sim_destroy", None, _vp)
    _declare(lib, "arc_sim_set_threads", None, _vp, ctypes.c_int)
    _declare(lib, "arc_sim_num_joints", ctypes.c_int)
    _declare(lib, "arc_sim_num_legs", ctypes.c_int)
    _declare(lib, "arc_sim_robot_info", None, _vp, _f64, _f64, _f64, _f64)
    _declare(lib, "arc_sim_reset", None, _vp, _i32, ctypes.c_int, _u32, _f64)
    # called every step: raw addresses, validated by the Sim methods, since
    # ndpointer argument checks cost a few microseconds per array
    _declare(lib, "arc_sim_step", None, _vp, _vp)
    _declare(lib, "arc_sim_observe", None, _vp, _vp, _vp, _vp, _vp, _vp)
    _declare(lib, "arc_sim_heights", None, _vp, _vp, ctypes.c_int, _vp)
    _declare(lib, "arc_sim_terrain_at", None, _vp, _i32, _f64, ctypes.c_int, _f64)
    return lib


@functools.cache
def render_lib() -> ctypes.CDLL:
    lib = _load("libarcrender.so")
    _declare(lib, "arc_render_create", _vp, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    _declare(lib, "arc_render_destroy", None, _vp)
    _declare(lib, "arc_render_frame", ctypes.c_int, _vp, _vp, ctypes.c_int, _vp,
             ctypes.c_int, ctypes.c_int, _vp)
    _declare(lib, "arc_render_size", None, _vp, ctypes.POINTER(ctypes.c_int),
             ctypes.POINTER(ctypes.c_int))
    _declare(lib, "arc_render_keys", ctypes.c_int, _vp, _i32, ctypes.c_int)
    _declare(lib, "arc_render_set_title", None, _vp, ctypes.c_char_p)
    _declare(lib, "arc_render_set_view", None, _vp, ctypes.c_int)
    return lib


class Sim:
    """N independent worlds, each with the quadruped on its own terrain."""

    def __init__(self, num_worlds: int, dt: float = 1.0 / 30.0, substeps: int = 20,
                 grid: int = 64, cell: float = 0.4, threads: int = 1):
        self._lib = sim_lib()
        self.num_worlds = num_worlds
        self.grid = grid
        self.cell = cell
        self.num_joints = self._lib.arc_sim_num_joints()
        self.num_legs = self._lib.arc_sim_num_legs()
        self._ptr = self._lib.arc_sim_create(num_worlds, dt, substeps, grid, cell)
        if not self._ptr:
            raise MemoryError("arc_sim_create failed")
        self._lib.arc_sim_set_threads(self._ptr, int(threads))

        nj = self.num_joints
        self.default_angles = np.zeros(nj)
        lo, hi, misc = np.zeros(nj), np.zeros(nj), np.zeros(2)  # limits, mass: unused here
        self._lib.arc_sim_robot_info(self._ptr, self.default_angles, lo, hi, misc)
        self.stand_height = float(misc[0])

        # output buffers, reused by observe()
        self.root = np.zeros((num_worlds, ROOT_SIZE))
        self.rot = np.zeros((num_worlds, 3, 3))
        self.joints = np.zeros((num_worlds, 3 * nj))
        self.contacts = np.zeros((num_worlds, self.num_legs + 2))
        self.flags = np.zeros(num_worlds, dtype=np.int32)
        self._observe_args = (self._ptr, self.root.ctypes.data, self.rot.ctypes.data,
                              self.joints.ctypes.data, self.contacts.ctypes.data,
                              self.flags.ctypes.data)

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def half_extent(self) -> float:
        return 0.5 * self.grid * self.cell

    def reset(self, ids: np.ndarray, seeds: np.ndarray, difficulty: np.ndarray) -> None:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        seeds = np.ascontiguousarray(seeds, dtype=np.uint32)
        difficulty = np.ascontiguousarray(difficulty, dtype=np.float64)
        assert len(ids) == len(seeds) == len(difficulty)
        if len(ids):
            self._lib.arc_sim_reset(self._ptr, ids, len(ids), seeds, difficulty)

    def step(self, targets: np.ndarray) -> None:
        targets = np.ascontiguousarray(targets, dtype=np.float64)
        assert targets.shape == (self.num_worlds, self.num_joints)
        self._lib.arc_sim_step(self._ptr, targets.ctypes.data)

    def observe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """root (N, 13), joints (N, 3*J: angles, rates, torques), contacts
        (N, legs + 2: feet, torso, other), unstable flags (N,). Also refreshes
        rot (N, 3, 3), the torso rotation matrices (body to world)."""
        self._lib.arc_sim_observe(*self._observe_args)
        return self.root, self.joints, self.contacts, self.flags

    def heights(self, offsets: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Terrain heights at (forward, left) offsets in each torso's heading frame."""
        offsets = np.ascontiguousarray(offsets, dtype=np.float64)
        if out is None:
            out = np.empty((self.num_worlds, len(offsets)))
        assert out.shape == (self.num_worlds, len(offsets)) and out.dtype == np.float64
        assert out.flags.c_contiguous
        self._lib.arc_sim_heights(self._ptr, offsets.ctypes.data, len(offsets), out.ctypes.data)
        return out

    def terrain_at(self, ids: np.ndarray, xy: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int32)
        xy = np.ascontiguousarray(xy, dtype=np.float64)
        out = np.zeros(len(ids))
        if len(ids):
            self._lib.arc_sim_terrain_at(self._ptr, ids, xy, len(ids), out)
        return out

    def close(self) -> None:
        if self._ptr:
            self._lib.arc_sim_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()


class Renderer:
    """Window ("human") or offscreen renderer showing one world of a Sim."""

    def __init__(self, width: int = 1280, height: int = 720, visible: bool = True):
        self._lib = render_lib()
        self._ptr = self._lib.arc_render_create(width, height, int(visible))
        if not self._ptr:
            raise RuntimeError("cannot create renderer (no display or no OpenGL 2.1?)")
        self.closed = False

    def size(self) -> tuple[int, int]:
        w, h = ctypes.c_int(), ctypes.c_int()
        self._lib.arc_render_size(self._ptr, ctypes.byref(w), ctypes.byref(h))
        return w.value, h.value

    def frame(self, sim: Sim, world: int = 0, markers: np.ndarray | None = None,
              active: int = -1, rgb: bool = False) -> np.ndarray | None:
        """Draws the world with checkpoint markers given as rows of (x, y, z,
        radius); returns an (H, W, 3) uint8 image if rgb is set."""
        img = None
        img_ptr = None
        if rgb:
            w, h = self.size()
            img = np.empty((h, w, 3), dtype=np.uint8)
            img_ptr = img.ctypes.data
        m_ptr, n = None, 0
        if markers is not None and len(markers):
            markers = np.ascontiguousarray(markers, dtype=np.float64).reshape(-1, 4)
            m_ptr, n = markers.ctypes.data, len(markers)
        if not self._lib.arc_render_frame(self._ptr, sim.ptr, world, m_ptr, n, active, img_ptr):
            self.closed = True
        return img

    def keys(self) -> list[int]:
        buf = np.zeros(32, dtype=np.int32)
        n = self._lib.arc_render_keys(self._ptr, buf, 32)
        return [int(k) for k in buf[:n]]

    def set_title(self, title: str) -> None:
        self._lib.arc_render_set_title(self._ptr, title.encode())

    def set_view(self, preset: int) -> None:
        """Camera preset around the robot's heading: 0 behind, 1 side, 2 front."""
        self._lib.arc_render_set_view(self._ptr, preset)

    def close(self) -> None:
        if self._ptr:
            self._lib.arc_render_destroy(self._ptr)
            self._ptr = None

    def __del__(self):
        self.close()
