"""Extension points for packs that build on SPEED.

Nothing here changes SPEED's own behaviour: with nothing registered, a run is the same as without
this module.

- ``register_noise_policy(name, start)`` adds a ``noise_policy``. ``start(noise, prepared, plan,
  stage_start_latent)`` returns ``(stage_start_noise, full_noise_coefficients)``: the first stage's
  noise in the latent's public layout, and either ``None`` (new bands at each transition come from
  seeded noise, like ``direct_coarse``) or DCT coefficients of a main-stream noise field at least as
  large as every stage (transitions take their new bands from it, like ``coupled_full_grid``).
- ``register_step_observer(factory)`` sees every sampling run of SPEED and Harvest. Each run calls
  ``factory(**meta)``; it returns ``None`` to skip the run, or an object with
  ``on_step(step, x0, x, *, stage, stage_sigmas, global_offset, nominal_sigmas)``, called on every
  sampler step before ComfyUI's own callback (``step`` counts within the stage, ``global_offset + step``
  across the run), and ``finish(final_latent)``. ``meta`` holds ``family``,
  ``sampler_path`` ("speed" or "native"), ``seed`` and ``sigmas``; SPEED runs add the stage plan
  (``noise_policy``, ``sigma_policy``, ``scales``, ``stage_shapes``, ``transition_steps``,
  ``full_shape``) and Harvest adds ``sampler_name``. An observer that raises is dropped for the rest
  of the run; sampling never fails because of one.
"""

from __future__ import annotations

import logging

from .config import NOISE_POLICIES

log = logging.getLogger(__name__)

_NOISE_POLICY_STARTS = {}
_OBSERVER_FACTORIES = []


def register_noise_policy(name: str, start) -> None:
    """Register ``start`` for ``noise_policy=name``. Registering the same function again is a no-op."""
    current = _NOISE_POLICY_STARTS.get(name)
    if current is not None and current is not start:
        raise ValueError(f"noise_policy {name!r} is already registered")
    _NOISE_POLICY_STARTS[name] = start
    NOISE_POLICIES.add(name)


def noise_policy_start(name: str):
    try:
        return _NOISE_POLICY_STARTS[name]
    except KeyError:
        raise ValueError(f"unsupported noise_policy: {name}") from None


def register_step_observer(factory) -> None:
    if factory not in _OBSERVER_FACTORIES:
        _OBSERVER_FACTORIES.append(factory)


def unregister_step_observer(factory) -> None:
    if factory in _OBSERVER_FACTORIES:
        _OBSERVER_FACTORIES.remove(factory)


class _Observers:
    """The observers of one run, each dropped after its first error."""

    def __init__(self, observers):
        self.observers = observers

    def _drop(self, observer, exc):
        log.warning("[SPEED] step observer %r failed (%r); dropped for this run", observer, exc)
        self.observers = [o for o in self.observers if o is not observer]

    def wrap(self, callback, **where):
        """``callback`` with every observer's ``on_step`` called first, for one stage."""
        if not self.observers:
            return callback

        def wrapped(step, x0, x, total_steps):
            for observer in list(self.observers):
                try:
                    observer.on_step(step, x0, x, **where)
                except Exception as exc:
                    self._drop(observer, exc)
            if callback is not None:
                return callback(step, x0, x, total_steps)

        return wrapped

    def finish(self, final):
        for observer in list(self.observers):
            try:
                observer.finish(final)
            except Exception as exc:
                self._drop(observer, exc)


def start_observers(**meta) -> _Observers:
    """Ask every registered factory for an observer of this run."""
    observers = []
    for factory in list(_OBSERVER_FACTORIES):
        try:
            observer = factory(**meta)
        except Exception as exc:
            log.warning("[SPEED] step observer factory %r failed (%r); skipped for this run", factory, exc)
            continue
        if observer is not None:
            observers.append(observer)
    return _Observers(observers)


__all__ = [
    "register_noise_policy",
    "noise_policy_start",
    "register_step_observer",
    "unregister_step_observer",
    "start_observers",
]
