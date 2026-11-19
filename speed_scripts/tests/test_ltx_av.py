"""LTX-2.x audio-video family: the video is resized, the audio stays on the video's clock.

The Advanced nodes (h3_video) detect LTX by its LTXAV latent format. The LTX errors are in test_families.py with the other families' errors.
"""

import pytest
import torch

from conftest import EchoGuider, NestedTensor, StubSampler, SeededNoise, ltx_patcher
from speed_scripts.config import SpeedConfig
from speed_scripts.families import LTXAVFamily, get_family
from speed_scripts.runtime import run_speed_pipeline

SIGMAS = torch.linspace(1.0, 0.0, 9)


def ltx_guider():
    guider = EchoGuider()
    guider.model_patcher = ltx_patcher()
    return guider


def ltx_latent(t=3, h=8, w=8):
    """Empty LTX AV latent: nested [1,128,T,H,W] video and [1,8,A,16] audio."""
    return {"samples": NestedTensor([torch.zeros(1, 128, t, h, w), torch.zeros(1, 8, 25, 16)])}


def test_advanced_nodes_detect_an_ltx_model():
    assert isinstance(get_family("h3_video", ltx_guider()), LTXAVFamily)


def test_ltx_run_resizes_video_and_keeps_the_audio_shape():
    guider = ltx_guider()
    config = SpeedConfig(scales=(.5, 1.0), transition_steps=(4,), transition_mode="explicit")
    out, _ = run_speed_pipeline(SeededNoise(3), guider, SIGMAS, ltx_latent(), config, sampler=StubSampler(),
                                model_type="h3_video", disable_pbar=True)
    video, audio = out["samples"].unbind()
    assert tuple(video.shape) == (1, 128, 3, 8, 8)
    assert tuple(audio.shape) == (1, 8, 25, 16)
    assert guider.latent_sizes == [(4, 4), (8, 8)]
    assert all(tuple(a.shape) == (1, 8, 25, 16) for _, a in guider.noise)


@pytest.mark.parametrize("policy", ["carry_preserve", "clock_reindex", "fresh"])
def test_ltx_audio_is_carried_unchanged(policy):
    audio = torch.randn(1, 8, 25, 16)
    assert LTXAVFamily(ltx_guider()).carry_extra(audio, .8, .6, policy, None) is audio


class MaskGuider(EchoGuider):
    """LTX EchoGuider that also records each stage's latent_image and denoise_mask."""

    def __init__(self):
        super().__init__()
        self.model_patcher = ltx_patcher()
        self.latents, self.masks = [], []

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.latents.append(latent_image.unbind())
        self.masks.append(kwargs.get("denoise_mask"))
        return super().sample(noise, latent_image, sampler, sigmas, callback=callback, **kwargs)


def i2v_latent(mask=True):
    """LTXVImgToVideoInplace + LTXVConcatAVLatent: the image in frame 0, pinned by the mask."""
    video = torch.zeros(1, 128, 3, 8, 8)
    video[:, :, 0] = torch.randn(1, 128, 8, 8, generator=torch.Generator().manual_seed(5))
    latent = {"samples": NestedTensor([video, torch.zeros(1, 8, 25, 16)])}
    if mask:
        video_mask = torch.ones(1, 1, 3, 1, 1)
        video_mask[:, :, 0] = 0.0
        latent["noise_mask"] = NestedTensor([video_mask, torch.ones(1, 8, 25, 16)])
    return latent


def _i2v_run(latent):
    guider = MaskGuider()
    config = SpeedConfig(scales=(.5, 1.0), transition_steps=(4,), transition_mode="explicit")
    run_speed_pipeline(SeededNoise(3), guider, SIGMAS, latent, config, sampler=StubSampler(),
                       model_type="h3_video", disable_pbar=True)
    return guider


def test_ltx_image_to_video_passes_the_mask_and_the_source_to_every_stage():
    latent = i2v_latent()
    guider = _i2v_run(latent)
    assert all(mask is latent["noise_mask"] for mask in guider.masks)
    (coarse, coarse_audio), (full, full_audio) = guider.latents
    assert tuple(coarse.shape[-2:]) == (4, 4) and torch.count_nonzero(coarse[:, :, 0])
    assert not torch.count_nonzero(coarse[:, :, 1:])
    assert torch.equal(full, latent["samples"].unbind()[0])
    assert torch.equal(full_audio, latent["samples"].unbind()[1])


def test_ltx_image_to_video_restarts_each_stage_on_the_carried_state():
    """Sampler start sigma*noise + (1-sigma)*source equals the unmasked run's start on the same state."""
    masked, plain = _i2v_run(i2v_latent()), _i2v_run(i2v_latent(mask=False))
    assert plain.masks == [None, None]
    sigma = plain.sigma_calls[1][0]
    start_masked = sigma * masked.noise[1][0] + (1 - sigma) * masked.latents[1][0]
    start_plain = sigma * plain.noise[1][0] + (1 - sigma) * plain.latents[1][0]
    assert torch.allclose(start_masked, start_plain, atol=1e-5)


def test_noise_masks_stay_unsupported_on_h3():
    latent = {"samples": NestedTensor([torch.zeros(1, 4, 3, 8, 8), torch.zeros(1, 8, 2, 25)]),
              "noise_mask": torch.ones(1, 1, 3, 1, 1)}
    config = SpeedConfig(scales=(.5, 1.0), transition_steps=(4,), transition_mode="explicit")
    with pytest.raises(ValueError, match="SPEED does not support noise masks"):
        run_speed_pipeline(SeededNoise(3), EchoGuider(), SIGMAS, latent, config, sampler=StubSampler(),
                           model_type="h3_video", disable_pbar=True)
