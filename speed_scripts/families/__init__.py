"""Model families SPEED can run, picked explicitly by the node, never auto-detected.

Each node class sets a default family with ``MODEL_TYPE``; the Advanced nodes
also take a ``model_type`` choice (``h3_video`` or ``ltx_av``). ``get_family``
checks the choice against the model's LTXAV latent format and raises with the
setting to use on a mismatch.

A family splits its latent into the spatial stream SPEED resizes (always
``[B, C, T, H, W]`` here) and an optional extra stream, reads a stage's output
back into sampler state, and carries the extra stream across a resolution
change. This module holds the registry and re-exports the family modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .h3_video import H3VideoFamily, pack_h3, resolve_sigma_shifts, unpack_h3
from .flow_image import FlowImageFamily, require_flow_sampling
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


def check_ltx_av_match(model_type, guider):
    """Raise when ``model_type`` and an LTX-2.x model (LTXAV latent format) disagree.

    Runs before the family is built, so an LTX model run as ``h3_video`` stops
    here rather than on the missing H3 sigma shifts.
    """
    ltx = is_ltx_av(guider)
    if ltx and model_type == "flow_image":
        raise ValueError(
            "SPEED KSampler nodes got an LTX-2 audio-video model (latent format LTXAV); "
            "use SPEED Advanced nodes with model_type ltx_av"
        )
    if ltx and model_type != "ltx_av":
        raise ValueError(
            f"model_type {model_type} got an LTX-2 audio-video model (latent format LTXAV); "
            "use model_type ltx_av"
        )
    if model_type == "ltx_av" and not ltx:
        raise ValueError(
            "model_type ltx_av needs an LTX-2 audio-video model (latent format LTXAV); "
            "use model_type h3_video for MiniMax-H3, or SPEED KSampler nodes for image models"
        )


def get_family(model_type, guider):
    """Build the family for a node's ``model_type`` (never auto-detected; a mismatch raises)."""
    check_ltx_av_match(check_model_type(model_type), guider)
    return _FAMILIES[model_type](guider)


@dataclass(frozen=True)
class PreparedLatent:
    """The family plus the fitted latent dict and its ``main``/``extra`` streams (``extra`` may be ``None``)."""

    family: H3VideoFamily | FlowImageFamily | LTXAVFamily
    latent: dict
    main: torch.Tensor
    extra: torch.Tensor | None


def prepare_family(model_type, guider, latent) -> PreparedLatent:
    """Pick the family, fit the latent and split it, checking it is unmasked.

    Only a family with ``accepts_source_latent`` takes a non-empty latent (image-to-image).
    """
    if "noise_mask" in latent:
        raise ValueError("SPEED does not support noise masks")
    family = get_family(model_type, guider)
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
    "LTXAVFamily",
    "get_family",
    "is_ltx_av",
    "unpack_ltx",
    "pack_h3",
    "PreparedLatent",
    "prepare_family",
    "prepare_latent",
    "require_flow_sampling",
    "resolve_sigma_shifts",
    "unpack_h3",
]
