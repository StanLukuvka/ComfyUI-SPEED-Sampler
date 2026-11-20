"""Extension points (speed_scripts/hooks.py and run_speed_pipeline's plan_fn) for packs built on SPEED."""

import importlib

import pytest
import torch

from conftest import EchoGuider, EchoNoise, SeededNoise, StubSampler, make_latent
from speed_scripts import hooks, runtime
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.planning import StagePlan
from speed_scripts.runtime import run_speed_pipeline

SIGMAS = torch.linspace(1.0, 0.0, 9)


def _cfg(**kwargs):
    return SpeedConfig(scales=(.5, 1.0), transition_steps=(4,), transition_mode="explicit", **kwargs)


def _run(config, guider=None, latent=None, noise=None, **kwargs):
    return run_speed_pipeline(
        noise or SeededNoise(), guider or EchoGuider(), SIGMAS, latent or make_latent(), config,
        sampler=StubSampler(), model_type="h3_video", disable_pbar=True, **kwargs,
    )


@pytest.fixture
def clean_hooks():
    policies = dict(hooks._NOISE_POLICY_STARTS)
    factories = list(hooks._OBSERVER_FACTORIES)
    yield
    for name in set(hooks._NOISE_POLICY_STARTS) - set(policies):
        hooks.NOISE_POLICIES.discard(name)
    hooks._NOISE_POLICY_STARTS.clear()
    hooks._NOISE_POLICY_STARTS.update(policies)
    hooks._OBSERVER_FACTORIES[:] = factories


def test_a_registered_noise_policy_starts_the_run_and_feeds_the_transitions(clean_hooks):
    calls = []

    def start(noise, prepared, plan, stage_start_latent):
        calls.append(plan.stage_shapes)
        h, w, t = plan.stage_shapes[0]
        main = torch.full((1, 1, t, h, w), 2.0)
        coefficients = torch.zeros(1, 1, 2, 8, 8)   # the transition's new bands come from here
        return prepared.family.join(main, torch.zeros_like(prepared.extra)), coefficients

    hooks.register_noise_policy("test_policy", start)
    guider = EchoGuider()
    _run(_cfg(noise_policy="test_policy"), guider)

    assert calls == [((4, 4, 2), (8, 8, 2))]
    assert torch.equal(guider.noise[0][0], torch.full((1, 1, 2, 4, 4), 2.0))
    assert guider.latent_sizes == [(4, 4), (8, 8)]


def test_noise_policies_cannot_be_registered_twice_under_one_name(clean_hooks):
    hooks.register_noise_policy("direct_coarse", runtime._start_direct_coarse)   # same function: fine
    with pytest.raises(ValueError, match="already registered"):
        hooks.register_noise_policy("direct_coarse", lambda *args: None)
    with pytest.raises(ValueError, match="unsupported noise_policy"):
        SpeedConfig(scales=(.5, 1.0), transition_steps=(4,), transition_mode="explicit", noise_policy="nope")


class _Recorder:
    def __init__(self, meta, fail=False):
        self.meta, self.fail, self.steps, self.final = meta, fail, [], None

    def on_step(self, step, x0, x, *, stage, stage_sigmas, global_offset, nominal_sigmas):
        if self.fail:
            raise RuntimeError("observer broke")
        self.steps.append((stage, global_offset + step))

    def finish(self, final):
        self.final = final


def test_step_observers_see_every_step_and_the_final_latent(clean_hooks):
    seen = []
    hooks.register_step_observer(lambda **meta: seen.append(_Recorder(meta)) or seen[-1])
    hooks.register_step_observer(lambda **meta: None)   # skips this run
    previews = []
    out, _ = _run(_cfg(), preview_callback=lambda step, x0, x, total: previews.append(step))

    (recorder,) = seen
    assert recorder.steps == [(0, i) for i in range(4)] + [(1, i) for i in range(4, 8)]
    assert previews == list(range(8))   # the stock callback still runs inside the wrapper
    assert recorder.final is out["samples"]
    assert recorder.meta["sampler_path"] == "speed" and recorder.meta["seed"] == 42
    assert recorder.meta["stage_shapes"] == [[4, 4, 2], [8, 8, 2]]
    assert recorder.meta["transition_steps"] == [4] and recorder.meta["noise_policy"] == "direct_coarse"


def test_a_failing_observer_never_changes_the_output(clean_hooks):
    plain, _ = _run(_cfg())
    previews = []
    hooks.register_step_observer(lambda **meta: _Recorder(meta, fail=True))   # on_step raises
    hooks.register_step_observer(lambda **meta: 1 / 0)                         # the factory raises
    observed, _ = _run(_cfg(), preview_callback=lambda step, x0, x, total: previews.append(step))
    assert previews == list(range(8))
    for a, b in zip(plain["samples"].unbind(), observed["samples"].unbind()):
        assert torch.equal(a, b)


def test_harvest_reports_to_observers_and_honours_the_noise_mask(clean_hooks):
    harvest_test = importlib.import_module("test_harvest_node")
    seen = []
    hooks.register_step_observer(lambda **meta: seen.append(_Recorder(meta)) or seen[-1])
    masks = []
    guider = harvest_test.HarvestGuider()
    sample = guider.sample
    guider.sample = lambda *a, **k: masks.append(k.get("denoise_mask")) or sample(*a, **k)
    latent = harvest_test._h3_latent()
    latent["noise_mask"] = mask = torch.ones(1)
    harvest_test._harvest(guider, latent=latent)

    assert masks == [mask]
    (recorder,) = seen
    assert recorder.meta["sampler_path"] == "native" and len(recorder.steps) == 19
    assert recorder.final is not None


def _shrinking_plan(config, sigmas, prepared):
    """A goal-style plan: start at 16x16 on an 8x8 latent, end at the latent's size."""
    return StagePlan(scales=(1.0, .5), stage_shapes=((16, 16, 2), (8, 8, 2)), transition_steps=(4,))


def test_plan_fn_replaces_the_stage_plan_and_may_step_down_a_size():
    guider = EchoGuider()
    out, _ = _run(_cfg(noise_policy="coupled_full_grid"), guider, plan_fn=_shrinking_plan,
                  noise=_WideNoise())
    assert guider.latent_sizes == [(16, 16), (8, 8)]
    assert out["samples"].unbind()[0].shape == (1, 1, 2, 8, 8)


class _WideNoise(SeededNoise):
    """Noise at the widest stage (16x16), as a goal plan's coupled field would be."""

    def generate_noise(self, latent):
        video, audio = latent["samples"].unbind()
        wide = {"samples": type(latent["samples"])([torch.zeros(*video.shape[:-2], 16, 16), audio])}
        return super().generate_noise(wide)


def test_a_source_image_refuses_a_plan_larger_than_the_latent():
    latent = make_latent()
    video, audio = latent["samples"].unbind()
    latent["samples"] = type(latent["samples"])([torch.ones_like(video), audio])
    with pytest.raises(ValueError, match="every stage fits the latent"):
        runtime._check_source_fits(video + 1, _shrinking_plan(None, None, None))
    runtime._check_source_fits(torch.zeros_like(video), _shrinking_plan(None, None, None))


def test_aligned_sigma_accepts_a_step_down():
    kappa, sigma = aligned_sigma(.5, .5)
    assert kappa == pytest.approx(.5 / .75) and sigma == pytest.approx(.5 * kappa)
    with pytest.raises(ValueError):
        aligned_sigma(.5, 1.0)


def test_coupled_transition_steps_down_by_dropping_the_top_bands():
    field = torch.randn(1, 1, 2, 16, 16)
    state = torch.randn(1, 1, 2, 16, 16)
    small = runtime._coupled_transition(state, field, (2, 8, 8), .3)
    expected = runtime.idct_temporal(runtime.idct2(
        runtime.dct2(runtime.dct_temporal(state))[..., :2, :8, :8]))
    assert torch.allclose(small, expected, atol=1e-5)


def test_families_report_pixels_per_latent_cell():
    from speed_scripts.families import h3_video, ltx_av
    assert h3_video.H3VideoFamily.spatial_factor == 16
    assert ltx_av.LTXAVFamily.spatial_factor == 32
