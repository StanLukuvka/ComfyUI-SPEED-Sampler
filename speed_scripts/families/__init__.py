"""Model families SPEED can run; each node class fixes its family with ``MODEL_TYPE``.

The exceptions are detected by latent format (``resolve_model_type``): the Advanced
nodes (``h3_video``) run ``ltx_av`` for LTX-2.x (LTXAV) and ``flow_image`` for
MiniMax-H3-Image (MiniMaxH3Video), so an H3-Image SamplerCustomAdvanced graph keeps
its own guider and sampler.

A family splits its latent into the spatial stream SPEED resizes (always
``[B, C, T, H, W]`` here) and an optional extra stream, reads a stage's output
back into sampler state, and carries the extra stream across a resolution
change. This module holds the registry and re-exports the family modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .h3_video import H3VideoFamily, pack_h3, resolve_sigma_shifts, unpack_h3
from .flow_image import FlowImageFamily, H3ImageFamily, is_h3_image, require_flow_sampling
from .ltx_av import LTXAVFamily, is_ltx_av, unpack_ltx

_FAMILIES = {"h3_video": H3VideoFamily, "flow_image": FlowImageFamily, "ltx_av": LTXAVFamily}
MODEL_TYPES = tuple(_FAMILIES)


def prepare_latent(guider, latent):
    """Copy of ``latent`` with the empty latent fitted to the model, as SamplerCustomAdvanced does.

    Generic empty-latent nodes (e.g. EmptyLatentImage) give 4 channels; ComfyUI widens
    them to the model's latent format. Nested (h3_video, ltx_av) latents pass through unchanged.
    """
    import comfy.sample

    latent = latent.copy()
    latent["samples"] = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    return latent


def check_model_type(model_type):
    """Raise unless ``model_type`` names a known family."""
    if model_type not in _FAMILIES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}; got {model_type!r}")
    return model_type


def resolve_model_type(model_type, guider):
    """The family to run: ``ltx_av`` for an LTX-2.x model (LTXAV latent format),
    ``flow_image`` for MiniMax-H3-Image on the Advanced nodes, else ``model_type``.

    The Advanced nodes (``h3_video``) pick up LTX and H3-Image by themselves; the
    KSampler nodes (``flow_image``) can't run LTX's nested latent and raise. Runs
    before the family is built, so neither model reaches the H3 sigma-shift lookup
    (H3-Image inherits H3's shift attributes but samples on its own sigma grid).
    """
    model_type = check_model_type(model_type)
    if model_type == "h3_video" and is_h3_image(guider):
        return "flow_image"
    if not is_ltx_av(guider):
        if model_type == "ltx_av":
            raise ValueError("ltx_av needs an LTX-2 audio-video model (latent format LTXAV)")
        return model_type
    if model_type == "flow_image":
        raise ValueError(
            "SPEED KSampler nodes got an LTX-2 audio-video model (latent format LTXAV); "
            "use the SPEED Advanced nodes (SPEED Sampler, Manual, Sigma Harvest)"
        )
    return "ltx_av"


def get_family(model_type, guider):
    """Build the family for a node's ``MODEL_TYPE``.

    An LTX-2.x model gets ``ltx_av``; MiniMax-H3-Image on the Advanced nodes gets
    ``flow_image`` as ``H3ImageFamily``, so its errors name the Advanced nodes.
    """
    resolved = resolve_model_type(model_type, guider)
    if model_type == "h3_video" and resolved == "flow_image":
        return H3ImageFamily(guider)
    return _FAMILIES[resolved](guider)


@dataclass(frozen=True)
class PreparedLatent:
    """The family plus the fitted latent dict and its ``main``/``extra`` streams (``extra`` may be ``None``)."""

    family: H3VideoFamily | FlowImageFamily | H3ImageFamily | LTXAVFamily
    latent: dict
    main: torch.Tensor
    extra: torch.Tensor | None


def prepare_family(model_type, guider, latent) -> PreparedLatent:
    """Pick the family, fit the latent and split it.

    Only a family with ``accepts_source_latent`` takes a non-empty latent (image-to-image),
    and only one with ``accepts_noise_mask`` takes a ``noise_mask`` (LTX image-to-video).
    """
    family = get_family(model_type, guider)
    if "noise_mask" in latent and not family.accepts_noise_mask:
        raise ValueError("SPEED does not support noise masks")
    latent = prepare_latent(guider, latent)

    main, extra = family.split(latent.get("samples"))
    empty = not torch.count_nonzero(main) and (extra is None or not torch.count_nonzero(extra))
    if not empty and not family.accepts_source_latent:
        raise ValueError(f"{family.nodes} require an empty input latent")
    return PreparedLatent(family=family, latent=latent, main=main, extra=extra)


__all__ = [
    "MODEL_TYPES",
    "H3VideoFamily",
    "FlowImageFamily",
    "H3ImageFamily",
    "LTXAVFamily",
    "get_family",
    "resolve_model_type",
    "is_ltx_av",
    "is_h3_image",
    "unpack_ltx",
    "pack_h3",
    "PreparedLatent",
    "prepare_family",
    "prepare_latent",
    "require_flow_sampling",
    "resolve_sigma_shifts",
    "unpack_h3",
]
