"""LTX-2.x audio-video family: the video is resized, the audio stays on the video's clock.

The model_type mismatch errors are in test_families.py with the other families' errors.
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


def test_model_type_ltx_av_builds_the_ltx_family():
    assert isinstance(get_family("ltx_av", ltx_guider()), LTXAVFamily)


def test_ltx_run_resizes_video_and_keeps_the_audio_shape():
    guider = ltx_guider()
    config = SpeedConfig(scales=(.5, 1.0), transition_steps=(4,), transition_mode="explicit")
    out, _ = run_speed_pipeline(SeededNoise(3), guider, SIGMAS, ltx_latent(), config, sampler=StubSampler(),
                                model_type="ltx_av", disable_pbar=True)
    video, audio = out["samples"].unbind()
    assert tuple(video.shape) == (1, 128, 3, 8, 8)
    assert tuple(audio.shape) == (1, 8, 25, 16)
    assert guider.latent_sizes == [(4, 4), (8, 8)]
    assert all(tuple(a.shape) == (1, 8, 25, 16) for _, a in guider.noise)


@pytest.mark.parametrize("policy", ["carry_preserve", "clock_reindex", "fresh"])
def test_ltx_audio_is_carried_unchanged(policy):
    audio = torch.randn(1, 8, 25, 16)
    assert LTXAVFamily(ltx_guider()).carry_extra(audio, .8, .6, policy, None) is audio
