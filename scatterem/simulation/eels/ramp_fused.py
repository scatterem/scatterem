"""The Fourier-shift ramp of ``_fourier_shift_stack`` as one Warp kernel.

Why this exists
---------------
``multislice_eels._fourier_shift_stack`` builds a complex64 phase ramp
``exp(-2j*pi*(ky*sy + kx*sx))`` for a batch of scan positions.  Written in eager
torch that is four elementwise passes, and each one writes a full
``(chunk, ny, nx)`` plane for the next one to read back:

    t     = ky*sy + kx*sx        write  8 B/element
    u     = (-2pi) * t           read 8  + write  8
    e     = polar(1, u)          read 8  + write 16   (complex128!)
    phase = e.to(complex64)      read 16 + write  8

= **72 B/element of traffic to produce 8 B/element of output**.  Measured at
``r7f_tds_adf/medium/probe_build`` (256x256 probe, 32 positions per call, 32
calls), D319's stage split:

    ramp: polar(1,u)        9.4395 ms   42.00 %
    ramp: .to(c64)          2.4834      11.05
    ramp: scale (-2pi)*t    1.7019       7.57
    ramp: add ky*sy+kx*sx   1.1175       4.97
                           -------      -----
    the ramp                14.74 ms    65.6 % of the row

One thread per ``(p, i, j)`` keeps the whole chain in registers and writes the
complex64 phase once.

What actually pays, and what does not
-------------------------------------
The traffic is **not** the prize, and that is the finding.  ``polar`` alone runs
at 168 GB/s against a 578 GB/s roof for its own 24 B/element -- it is bound by
**double-precision ``sin``/``cos``**, not by memory: measured, ``torch.cos(u)``
alone is 0.1556 ms and ``torch.sin(u)`` alone 0.1546, and ``polar`` is 0.300,
i.e. exactly their sum.  So a fused kernel that merely deletes the other three
passes buys almost nothing -- measured at **1.029x** on the region, because the
5.30 ms of traffic it removes is small beside the 9.44 ms of transcendentals it
still has to do.

What pays is doing the two transcendentals **together**.  ``sin(u)`` and
``cos(u)`` as separate calls each perform their own Payne-Hanek argument
reduction, and at the angles this path uses (``|u|`` up to 804 at the medium
config, because the shifts are hundreds of pixels) that reduction is the
expensive half.  CUDA's ``sincos()`` shares it, and the result is bit-identical
to calling both separately -- verified, not assumed.  **1.296x on the region.**

Bit-exactness is a contract here, not an aspiration
---------------------------------------------------
``r7f_tds_adf`` and ``r7_eels`` are both gated on ``sha256`` of their output, and
this helper's own docstring in ``multislice_eels`` records two rewrites that were
rejected for moving bits (a float32 ramp, and factoring the ramp into a separable
outer product).  Three spellings below are load-bearing and each was measured
against torch on the real operands rather than reasoned about:

* **the association is ``(-2pi) * ((ky*sy) + (kx*sx))``**, exactly as the eager
  body writes it -- not ``(-2pi*ky)*sy + ...``.  Multiplication by ``-2pi`` does
  not distribute over a rounded sum.  Measured: the redistributed spelling is
  *nearly* exact and differs on 1 of 144 gate regimes (``max|d|`` 2.4e-07, at
  +-2000 px shifts) -- which is exactly the failure mode that makes the
  separable ramp untakeable, arriving here one step earlier.
* **``sincos(a, &sin, &cos)`` == ``sin(a)``, ``cos(a)``** on this toolkit: 0
  differing float32 words of 4 194 304 over the real medium-config angles, and 0
  over the 144-regime battery in ``benchmarks/lab/_probe_d319_gate.py``.
* **``__dmul_rn`` / ``__dadd_rn`` are a GUARANTEE here, not a demonstrated
  necessity, and the difference is worth stating.**
  ``diffraction_tomo/kinematic_fused.py`` records that nvrtc contracts
  ``a*b + c*d`` into an FMA -- true, measured, and load-bearing *there* (16103 of
  65536 float32 values differ).  For **this** kernel it is FALSE on this
  toolchain: the mutant spelled with plain ``*`` and ``+`` passes the whole
  144-regime battery, 0 differing words of 9 383 936.  The intrinsics are kept
  because they cost nothing and make the property independent of a compiler flag
  no caller controls -- but do not repeat the "measured to be load-bearing"
  claim about them without re-measuring it.

Scope of the gate
-----------------
CUDA only (``wp.launch`` on a CPU device is a single-threaded scalar loop, so the
fusion *loses* there -- see the perf-lab D302 finding), complex64 output only
(complex128 callers keep the eager path, which is already only 24 B/element of
downcast traffic they do not pay), and forward-only: the kernel has no adjoint,
so a caller differentiating w.r.t. the scan positions keeps the eager chain and
no gradcheck obligation arises.
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

# ``a * b`` and ``a + b`` with the round-to-nearest intrinsics, so nvrtc cannot
# contract the ramp into an FMA.  See the module docstring.
_MUL_SRC = "return __dmul_rn(a, b);"
_ADD_SRC = "return __dadd_rn(a, b);"

# One shared argument reduction for the pair.  ``sincos`` writes sin then cos;
# the vector is returned as (cos, sin) to match the complex layout (re, im).
_SINCOS_SRC = (
    "double2 r; sincos(a, &r.y, &r.x); "
    "return wp::vec_t<2, wp::float64>(r.x, r.y);"
)


@wp.func_native(_MUL_SRC)
def _dmul(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_ADD_SRC)
def _dadd(a: wp.float64, b: wp.float64) -> wp.float64: ...


@wp.func_native(_SINCOS_SRC)
def _dsincos(a: wp.float64) -> wp.vec2d: ...


@wp.kernel
def _ramp_phase_c64_kernel(
    ky: wp.array(dtype=wp.float64),
    kx: wp.array(dtype=wp.float64),
    sy: wp.array(dtype=wp.float64),
    sx: wp.array(dtype=wp.float64),
    minus_two_pi: wp.float64,
    out: wp.array4d(dtype=wp.float32),
):
    p, i, j = wp.tid()
    t = _dadd(_dmul(ky[i], sy[p]), _dmul(kx[j], sx[p]))
    u = _dmul(minus_two_pi, t)
    cs = _dsincos(u)
    out[p, i, j, 0] = wp.float32(cs[0])
    out[p, i, j, 1] = wp.float32(cs[1])


_KWP: dict = {}


def _wp_freqs(k: Tensor) -> "wp.array":
    """``wp.from_torch`` of a ``_fftfreq_f64`` plane, memoised per (n, device).

    ``ky``/``kx`` come from ``multislice_eels._fftfreq_f64``'s own memo, so they
    are the *same objects* on every call of a scan; converting them costs host
    time per ``wp.from_torch`` and buys nothing (perf-lab QD63).  Keyed on the
    tensor's identity so a different memo entry can never alias this one.
    """
    key = (id(k), int(k.shape[0]), str(k.device))
    v = _KWP.get(key)
    if v is None:
        v = wp.from_torch(k.contiguous(), dtype=wp.float64)
        _KWP[key] = (v, k)
        return v
    return v[0]


def ramp_phase_fused_supported(waves: Tensor, shifts: Tensor) -> bool:
    """Whether the fused kernel reproduces the eager ramp for these operands.

    Everything here is host-known, so the predicate costs no device sync.  The
    excluded cases are the ones this kernel does not carry: a CPU device (Warp
    runs one scalar thread there), a non-complex64 phase dtype, and a caller
    differentiating w.r.t. the scan positions -- ``shifts`` is the only operand
    of the ramp, and the kernel has no adjoint.
    """
    return (
        waves.device.type == "cuda"
        and waves.dtype == torch.complex64
        and not (torch.is_grad_enabled() and shifts.requires_grad)
    )


def ramp_phase_fused(
    ky: Tensor,
    kx: Tensor,
    sy: Tensor,
    sx: Tensor,
    minus_two_pi: float,
    shape: tuple,
) -> Tensor:
    """``polar(1, (-2pi)*(ky*sy + kx*sx)).to(complex64)``, one kernel.

    ``ky``/``kx`` are 1-D ``(ny,)`` / ``(nx,)``; ``sy``/``sx`` are 1-D ``(P,)``.
    ``shape`` is the caller's broadcast phase shape, whose trailing two axes are
    ``(ny, nx)`` and whose leading axes multiply to ``P`` -- the middle axes are
    all 1, so the output is addressed as ``(P, ny, nx)``.
    """
    ny = int(ky.shape[0])
    nx = int(kx.shape[0])
    P = int(sy.shape[0])
    phase = torch.empty(shape, dtype=torch.complex64, device=ky.device)
    dst = torch.view_as_real(phase).reshape(P, ny, nx, 2)
    wp.launch(
        _ramp_phase_c64_kernel,
        dim=(P, ny, nx),
        inputs=[
            _wp_freqs(ky),
            _wp_freqs(kx),
            wp.from_torch(sy.contiguous(), dtype=wp.float64),
            wp.from_torch(sx.contiguous(), dtype=wp.float64),
            wp.float64(minus_two_pi),
            wp.from_torch(dst, dtype=wp.float32),
        ],
        device=wp.device_from_torch(ky.device),
        # NOT a detail: Warp's default stream for a device is not PyTorch's
        # current stream, so a launch left on Warp's own stream is unordered
        # against the ATen ops that consume ``phase`` and is INVISIBLE to a
        # ``torch.cuda.graph`` capture -- on replay the phase is simply never
        # written and the result is uninitialised memory.  Measured, not
        # reasoned about: without this the r7f_tds_adf oracle reports 103
        # failures with NaNs and rel 7.5e+01.  Same rule as
        # ``simulation/tds_emission_fused.py``.
        stream=wp.stream_from_torch(torch.cuda.current_stream(ky.device)),
    )
    return phase
