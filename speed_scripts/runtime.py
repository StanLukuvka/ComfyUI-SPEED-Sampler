"""Run SPEED across several resolution stages.

Each stage is one ``guider.sample()`` call. Between stages the state grows to
the next size and continues from the aligned sigma. Latent layout comes from
the model family.
"""

from __future__ import annotations

import logging

import torch

from .config import SpeedConfig
from .families import PreparedLatent, prepare_family
from .hooks import noise_policy_start, register_noise_policy, start_observers
from .flow import aligned_sigma, reentry_noise
from .latent_class import LatentWalker
from .planning import StagePlan, plan_stages
from .spectral import (
    dct2,
    dct_temporal,
    idct2,
    idct_temporal,
    lowpass_dct,
    spectral_expand,
    spectral_expand_3d,
)

log = logging.getLogger(__name__)


def _build_preview_callback(guider, total_steps, x0_output):
    """Build ComfyUI's preview callback."""
    try:
        import latent_preview
    except Exception as exc:
        log.info("[SPEED-preview] latent_preview unavailable (%r) — previews disabled", exc)
        return None
    return latent_preview.prepare_callback(guider.model_patcher, total_steps, x0_output)


class _PreviewGuard:
    """Run-wide preview state, so a failing preview stays disabled across stages and warns once."""

    def __init__(self):
        self.disabled = False

    def callback(self, stock_cb, capture_state, global_offset, global_total):
        """Bind one stage-local callback onto the run-wide progress timeline."""
        if stock_cb is None:
            try:
                import comfy.utils as comfy_utils

                pbar = comfy_utils.ProgressBar(global_total)
            except Exception:
                pbar = None

            def callback(step, x0, x, total_steps):
                capture_state["x0"] = x0
                if pbar is not None:
                    try:
                        pbar.update_absolute(global_offset + step + 1, global_total)
                    except Exception:
                        pass

            return callback

        def callback(step, x0, x, total_steps):
            capture_state["x0"] = x0
            if self.disabled:
                return
            try:
                stock_cb(global_offset + step, x0, x, global_total)
            except Exception as exc:
                self.disabled = True
                log.warning(
                    "[SPEED-preview] preview callback failed (%r) — disabling updates for this run",
                    exc,
                )

        return callback


def _coupled_transition(
    internal_main,
    full_noise_coefficients,
    target_thw,
    sigma: float,
):
    """Move the stage to ``target_thw`` on the shared noise field: shrinking drops the top bands."""
    target_t, target_h, target_w = target_thw
    source_t, source_h, source_w = internal_main.shape[-3:]
    shared_t, shared_h, shared_w = min(source_t, target_t), min(source_h, target_h), min(source_w, target_w)
    full_noise_coefficients = full_noise_coefficients.to(device=internal_main.device)
    target_coefficients = (
        full_noise_coefficients[..., :target_t, :target_h, :target_w].clone()
        * float(sigma)
    )
    target_coefficients[..., :shared_t, :shared_h, :shared_w] = dct2(dct_temporal(internal_main))[
        ..., :shared_t, :shared_h, :shared_w
    ]
    return idct_temporal(idct2(target_coefficients)).to(dtype=internal_main.dtype)


def _downsample_source(full_main, target_thw):
    """First-stage source latent: zeros for an empty latent, else the image low-passed to the stage size.

    The DCT low-pass is rescaled to keep the mean, so the coarse image has the
    same level as the full one; a per-channel affine latent format commutes with it.
    """
    target_t, target_h, target_w = target_thw
    shape = full_main.shape[:-3] + (target_t, target_h, target_w)
    if not torch.count_nonzero(full_main):
        return full_main.new_zeros(shape)
    if target_t != full_main.shape[-3]:
        raise ValueError("SPEED image-to-image does not support temporal stage scales")
    source_h, source_w = full_main.shape[-2:]
    if (target_h, target_w) == (source_h, source_w):
        return full_main
    gain = ((target_h * target_w) / (source_h * source_w)) ** 0.5
    return lowpass_dct(full_main, (target_h, target_w)) * gain


def _check_source_fits(full_main, plan):
    """Raise before sampling when a source image (image-to-image, LTX image-to-video) would need a
    stage larger than the latent: the source can only be low-passed down. SPEED's own plans never
    do this; a plan from ``run_speed_pipeline(plan_fn=...)`` can."""
    if not torch.count_nonzero(full_main):
        return
    source_h, source_w = full_main.shape[-2:]
    if any(h > source_h or w > source_w for h, w, _ in plan.stage_shapes):
        raise ValueError(
            "a source image (image-to-image or LTX image-to-video) only works when every stage fits the "
            f"latent ({source_h}x{source_w} cells); this plan has a larger stage"
        )


def _start_direct_coarse(noise, prepared, plan, stage_start_latent):
    coarse_latent = prepared.latent.copy()
    coarse_latent["samples"] = stage_start_latent
    return noise.generate_noise(coarse_latent), None


def _start_coupled_full_grid(noise, prepared, plan, stage_start_latent):
    family = prepared.family
    stage_h, stage_w, stage_t = plan.stage_shapes[0]
    full_noise_main, full_noise_extra = family.split(noise.generate_noise(prepared.latent))
    coarse_noise_main = lowpass_dct(full_noise_main[..., :stage_t, :, :], (stage_h, stage_w))
    full_noise_coefficients = dct2(dct_temporal(full_noise_main))
    return family.join(coarse_noise_main, full_noise_extra), full_noise_coefficients


register_noise_policy("direct_coarse", _start_direct_coarse)
register_noise_policy("coupled_full_grid", _start_coupled_full_grid)


def _expand_main(
    internal_main,
    *,
    target_thw,
    sigma: float,
    seed: int,
    full_noise_coefficients=None,
):
    """Move the carried spatial state to the next stage size (larger, or smaller under a custom plan)."""
    target_t, target_h, target_w = target_thw
    if full_noise_coefficients is not None:
        return _coupled_transition(
            internal_main,
            full_noise_coefficients,
            target_thw,
            sigma,
        )
    source_h, source_w = internal_main.shape[-2:]
    if target_h < source_h or target_w < source_w:   # a smaller stage keeps the low bands
        internal_main = lowpass_dct(internal_main, (min(source_h, target_h), min(source_w, target_w)))
    if target_t > internal_main.shape[-3]:
        return spectral_expand_3d(
            internal_main,
            target_thw,
            sigma,
            seed,
        )
    return spectral_expand(
        internal_main,
        (target_h, target_w),
        sigma,
        seed,
    )


def run_speed_pipeline(
    noise,
    guider,
    sigmas: torch.Tensor,
    latent: dict,
    config: SpeedConfig,
    *,
    sampler,
    model_type: str,
    disable_pbar: bool = False,
    output_device=None,
    preview_callback=None,
    x0_output=None,
    plan_fn=None,
):
    """Run ``prepare_family``, ``plan_stages``, then ``run_stages``.

    ``sampler`` is any ComfyUI SAMPLER. ``model_type`` picks the family. ``plan_fn(config, sigmas,
    prepared)`` replaces ``plan_stages`` and returns a ``StagePlan``; its stages may be larger or
    smaller than the latent, and the run ends at the latent's size.
    """
    prepared = prepare_family(model_type, guider, latent)
    if plan_fn is not None:
        plan = plan_fn(config, sigmas, prepared)
    else:
        plan = plan_stages(config, sigmas, prepared.main.shape)
    if plan_fn is None and len(plan.scales) < len(config.scales):
        log.info(
            "[SPEED] sigma schedule starts at %.4f, below the first %d stage threshold(s): running %d stage(s)",
            float(sigmas[0]),
            len(config.scales) - len(plan.scales),
            len(plan.scales),
        )
    return run_stages(
        noise,
        guider,
        sigmas,
        prepared,
        config,
        plan,
        sampler=sampler,
        disable_pbar=disable_pbar,
        output_device=output_device,
        preview_callback=preview_callback,
        x0_output=x0_output,
    )


def run_stages(
    noise,
    guider,
    sigmas: torch.Tensor,
    prepared: PreparedLatent,
    config: SpeedConfig,
    plan: StagePlan,
    *,
    sampler,
    disable_pbar: bool = False,
    output_device=None,
    preview_callback=None,
    x0_output=None,
):
    """Run the planned stages over the prepared latent."""
    family = prepared.family
    latent = prepared.latent
    full_main = prepared.main
    full_extra = prepared.extra
    scales = plan.scales
    stage_shapes = plan.stage_shapes
    transition_steps = plan.transition_steps

    _check_source_fits(full_main, plan)
    working_sigmas = sigmas.clone()
    stage_h, stage_w, stage_t = stage_shapes[0]
    coarse_main = _downsample_source(full_main, (stage_t, stage_h, stage_w))
    # A noise_mask (LTX image-to-video) pins the masked frames to the source at every step, so
    # every stage gets the mask and the source at its own size; ComfyUI resizes the mask.
    noise_mask = latent.get("noise_mask")
    source_extra = full_extra if noise_mask is not None else _maybe(torch.zeros_like, full_extra)
    stage_start_latent = family.join(coarse_main, source_extra)

    start_noise = noise_policy_start(config.noise_policy)
    stage_start_pub, full_noise_coefficients = start_noise(noise, prepared, plan, stage_start_latent)

    if x0_output is None:
        x0_output = {}
    global_total = len(sigmas) - 1
    stock_callback = preview_callback
    if stock_callback is None:
        stock_callback = _build_preview_callback(guider, global_total, x0_output)
    preview_guard = _PreviewGuard()
    observers = start_observers(
        family=family,
        sampler_path="speed",
        seed=int(noise.seed),
        sigmas=[float(s) for s in sigmas],
        noise_policy=config.noise_policy,
        sigma_policy=config.sigma_policy,
        scales=list(scales),
        stage_shapes=[list(shape) for shape in stage_shapes],
        transition_steps=list(transition_steps),
        full_shape=list(full_main.shape),
    )

    reference_hw = tuple(full_main.shape[-2:]) if family.scales_reference_latents else None
    walker = LatentWalker(guider, reference_hw)
    global_start = 0
    global_done = 0

    try:
        for stage_idx, global_end in enumerate(transition_steps):
            stage_h, stage_w, _ = stage_shapes[stage_idx]
            walker.apply_stage(stage_h, stage_w)

            stage_sigmas = working_sigmas[global_start : global_end + 1]
            callback = preview_guard.callback(
                stock_callback,
                x0_output,
                global_done,
                global_total,
            )
            callback = observers.wrap(
                callback, stage=stage_idx, stage_sigmas=stage_sigmas, global_offset=global_done, nominal_sigmas=sigmas
            )
            public = guider.sample(
                stage_start_pub,
                stage_start_latent,
                sampler,
                stage_sigmas,
                denoise_mask=noise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
            global_done += len(stage_sigmas) - 1

            boundary_sigma = float(working_sigmas[global_end])
            internal_main, internal_extra = family.to_internal(public, boundary_sigma)

            ratio = scales[stage_idx + 1] / scales[stage_idx]
            if config.sigma_policy == "canonical":
                kappa, next_sigma = aligned_sigma(boundary_sigma, ratio)
            else:
                kappa, next_sigma = 1.0, boundary_sigma
            working_sigmas[global_end] = next_sigma

            next_h, next_w, next_t = stage_shapes[stage_idx + 1]
            expanded_main = _expand_main(
                internal_main,
                target_thw=(next_t, next_h, next_w),
                sigma=boundary_sigma,
                seed=int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                full_noise_coefficients=full_noise_coefficients,
            )
            transitioned_main = expanded_main * kappa
            transitioned_extra = family.carry_extra(
                internal_extra,
                boundary_sigma,
                next_sigma,
                config.audio_policy,
                x0_output.get("x0"),
            )

            if noise_mask is None:
                stage_start_pub = family.join(
                    reentry_noise(transitioned_main, next_sigma),
                    _maybe(lambda state: reentry_noise(state, next_sigma), transitioned_extra),
                )
                stage_start_latent = family.join(
                    torch.zeros_like(transitioned_main),
                    _maybe(torch.zeros_like, transitioned_extra),
                )
            else:
                # The sampler starts from sigma * noise + (1 - sigma) * source; remove the
                # source term so the stage starts exactly on the carried state.
                source_main = _downsample_source(full_main, (next_t, next_h, next_w))
                stage_start_pub = family.join(
                    reentry_noise(transitioned_main - (1.0 - next_sigma) * source_main.to(transitioned_main),
                                  next_sigma),
                    _maybe(lambda state: reentry_noise(state - (1.0 - next_sigma) * full_extra.to(state), next_sigma),
                           transitioned_extra),
                )
                stage_start_latent = family.join(source_main, full_extra)
            global_start = global_end

        walker.apply_final()
        final_sigmas = working_sigmas[global_start:]
        final_callback = preview_guard.callback(
            stock_callback,
            x0_output,
            global_done,
            global_total,
        )
        final_callback = observers.wrap(
            final_callback,
            stage=len(transition_steps),
            stage_sigmas=final_sigmas,
            global_offset=global_done,
            nominal_sigmas=sigmas,
        )
        last_public = guider.sample(
            stage_start_pub,
            stage_start_latent,
            sampler,
            final_sigmas,
            denoise_mask=noise_mask,
            callback=final_callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        observers.finish(last_public)

        if output_device is not None:
            last_public = last_public.to(output_device)

        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = last_public

        denoised = output
        x0 = x0_output.get("x0")
        if x0 is not None:
            denoised = latent.copy()
            denoised["samples"] = guider.model_patcher.model.process_latent_out(
                x0.cpu() if hasattr(x0, "cpu") else x0
            )
        return output, denoised
    finally:
        walker.apply_final()


def _maybe(fn, value):
    return None if value is None else fn(value)
