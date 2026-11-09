from __future__ import annotations

import torch
from torch import Size, Tensor
from torch.nn import Module

from scatterem.physics._frequency_grid import fft_frequency_grid


# Not physics.fourier.fourier_shift: fp32 ramp buffers feed the compiled step.
class SubpixShift(Module):
    """Fourier-domain sub-pixel shift of a batch of probe wave fields.

    Shifts each wave by a per-position fractional offset ``dr`` via a linear
    phase ramp applied in the Fourier domain, so probe positions need not
    land on integer detector pixels.

    Parameters
    ----------
    probe_shape : tuple or torch.Size
        Probe tensor shape ``(a, b, MY, MX)``; only the detector-plane
        dimensions ``(MY, MX)`` are used to build the frequency ramps.
    device : str or torch.device or None
        Device to build the frequency-ramp buffers on.
    subpix_shift : bool
        Whether ``forward`` applies the sub-pixel shift; if False, ``forward``
        only broadcasts ``waves`` to the batch of positions.
    rdtype : torch.dtype, optional
        Real dtype of the stored frequency ramps. Default is ``torch.float32``.
    """

    def __init__(
        self,
        probe_shape: tuple | Size,
        device: str | torch.device | None,
        subpix_shift: bool,
        rdtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        a, b, MY, MX = probe_shape
        q = fft_frequency_grid((MY, MX), [1, 1], device=device)
        # Store the frequency ramps at the working real precision (float16 in
        # fp16 mode) so the forward needs no per-call cast.
        # Registered buffers, so `.to(device)` moves the ramps with the module;
        # as plain attributes they stayed on the construction device and the
        # forward's `qqx * dr` then raised on a cross-device multiply.
        # NON-persistent: both are fully determined by `probe_shape`/`rdtype`,
        # so they do not belong in `state_dict()` and this does not change what
        # a checkpoint contains.
        self.register_buffer("qqy", q[0][None, None, ...].to(rdtype), persistent=False)
        self.register_buffer("qqx", q[1][None, None, ...].to(rdtype), persistent=False)
        self.subpix_shift = subpix_shift
        # May the forward move the round-trip's normalization off the broadcast
        # end (see forward)? Only where the move is BIT-IDENTICAL: the swap is
        # "ortho then ortho" -> "1/N then bare", so 1/sqrt(N) must itself be an
        # exact binary scaling, i.e. N = MY*MX must be a power of FOUR. That is
        # a strictly stronger condition than Propagator.forward's `non_pow2`
        # (which folds 1/N and so only needs N a power of two) -- a 128x256
        # grid has N = 2**15 and is excluded here but not there. Do not
        # transfer that predicate between the two sites.
        n = int(MY) * int(MX)
        self._fold_norm = n > 0 and (n & (n - 1)) == 0 and (n.bit_length() - 1) % 2 == 0

    def forward(self, waves: Tensor, dr: Tensor) -> Tensor:
        """Apply a per-position sub-pixel shift and broadcast over positions.

        Parameters
        ----------
        waves : torch.Tensor
            Complex probe wave field(s), shape (Nmodes, K, MY, MX) or
            (Nmodes, 1, MY, MX).
        dr : torch.Tensor
            Fractional (row, col) shift per position, shape (K, 2).

        Returns
        -------
        torch.Tensor
            Shifted (or broadcast) complex wave field, shape
            (Nmodes, K, MY, MX).
        """
        # ramp is shape 1 x K x MY x MX
        if self.subpix_shift:
            rdtype = self.qqx.dtype
            rampx = self.qqx * dr[:, 1][:, None, None].to(rdtype)
            rampy = self.qqy * dr[:, 0][:, None, None].to(rdtype)
            # ramp follows waves' dtype (robust against a dtype mismatch) so a
            # complex32 wave field is never silently upcast.
            ramp = torch.exp((-2j * torch.pi) * (rampx + rampy)).to(waves.dtype)
            # The round-trip's total scaling is 1/N however `norm` splits it, and
            # the two ends are NOT the same size: `waves` is the un-broadcast
            # probe [Nmodes,1,MY,MX] while `Psi * ramp` has broadcast over K
            # ([Nmodes,K,MY,MX], 256x larger at a 256-position batch). "ortho"
            # therefore pays half the normalization as a full pass over the BIG
            # end. Put the whole 1/N on the small end instead (`norm="forward"`
            # on the forward transform = 1/N) and take the UNNORMALIZED inverse
            # (`norm="forward"` on an inverse transform = no scaling), which
            # removes that pass. Same fold as Propagator.forward above.
            # `is_cuda`: torch applies `norm=` as a separate exact post-scaling
            # only on the CUDA backend; CPU (pocketfft) folds the factor INTO
            # the transform, so the same move changes the last ulp there and
            # breaks the bit-exactness contracts in tests/. Not for complex32
            # either: 1/N instead of 1/sqrt(N) pushes the intermediate towards
            # fp16 subnormals, where the scaling stops being exact.
            if self._fold_norm and waves.is_cuda and waves.dtype != torch.complex32:
                Psi = torch.fft.fft2(waves, norm="forward")
                inv_norm = "forward"
            else:
                Psi = torch.fft.fft2(waves, norm="ortho")
                inv_norm = "ortho"
            # B x K x MY x MX
            # print(ramp.shape,Psi.shape)
            if len(Psi.shape) < 4:
                Psi = Psi.unsqueeze(1)
                Psi = Psi.repeat(1, ramp.shape[1], 1, 1)
            # B x K x MY x MX
            w = torch.fft.ifft2(Psi * ramp, norm=inv_norm)
        else:
            ps = waves.shape
            w = waves.expand((ps[0], dr.shape[0], ps[2], ps[3]))
        return w
