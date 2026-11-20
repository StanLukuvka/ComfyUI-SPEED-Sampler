"""LTX-2.x audio-video (``ltx_av``) model family.

The latent is a nested video/audio pair, as for MiniMax-H3. LTX conditions the audio on the same
timestep as the video, so the audio stays on the video's sigma clock: it is carried unchanged across
a resolution change. The latent format is the identity (scale 1, no mean).
"""

from __future__ import annotations

import torch

from .h3_video import pack_h3, to_internal_state
from .flow_image import require_flow_sampling


def is_ltx_av(guider):
    """True when the guider's model uses ComfyUI's LTXAV latent format."""
    try:
        latent_format = guider.model_patcher.get_model_object("latent_format")
    except AttributeError:  # a guider or patcher stand-in without the lookup
        return False
    return type(latent_format).__name__ == "LTXAV"


def unpack_ltx(samples):
    """Validate and unpack the LTX nested latent into ``(video, audio)``."""
    if not getattr(samples, "is_nested", False):
        raise ValueError("LTX-2 needs the nested video/audio latent (LTXVConcatAVLatent)")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError("LTX-2 latents need exactly video and audio streams")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("LTX-2 expects video [B,C,T,H,W] and audio [B,C,T,F]")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("LTX-2 supports batch size one")
    return video, audio


class LTXAVFamily:
    """LTX-2.x: resize the video stream, carry the audio stream on the same sigma clock."""

    name = "ltx_av"
    nodes = "SPEED Advanced nodes"
    # Image-to-video (LTXVImgToVideo*, LTXVAddGuide): the image frames are written into the
    # latent and pinned by its noise_mask, which ComfyUI resizes to each stage's latent.
    accepts_source_latent = True
    accepts_noise_mask = True
    scales_reference_latents = False
    spatial_factor = 32   # pixels per latent cell

    def __init__(self, guider):
        require_flow_sampling(guider, "SPEED Advanced nodes on LTX-2")

    def split(self, samples):
        return unpack_ltx(samples)

    def join(self, main, extra):
        return pack_h3(main, extra)

    def main_stream(self, value):
        """Best-effort ``[B, C, T, H, W]`` spatial stream for diagnostics, or ``None``."""
        if not getattr(value, "is_nested", False):
            return None
        value = next((s for s in value.unbind() if getattr(s, "ndim", 0) == 5), None)
        return value if isinstance(value, torch.Tensor) else None

    def to_internal(self, public, sigma):
        # Identity latent format: ComfyUI hands back x / (1 - sigma) for both streams.
        video, audio = self.split(public)
        return to_internal_state(video, audio, sigma, 1.0)

    def carry_extra(self, audio, old_sigma, new_sigma, policy, x0):
        # One sigma clock: every audio policy keeps the audio state as it is.
        return audio
