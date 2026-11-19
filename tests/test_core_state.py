from __future__ import annotations

import pytest

from deeptutor.learning.models import (
    KnowledgePoint,
    LearningModule,
    LearningProgress,
    RepetitionState,
)
from deeptutor.review import core_state, storage as storage_module
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import AttemptInput, ReviewService

NOW = 1_800_000_000.0
CORE_DUE = NOW - 120.0


def _progress(*, with_state: bool = True) -> LearningProgress:
    progress = LearningProgress(
        book_id="fixture-mastery",
        desired_retention=0.9,
        modules=[
            LearningModule(
                id="m1",
                name="module",
                order=1,
                knowledge_points=[
                    KnowledgePoint(
                        id="kp-series-resonance",
                        name="串联谐振",
                        type="concept",
                        module_id="m1",
                    ),
                    KnowledgePoint(
                        id="kp-quality-factor",
                        name="品质因数",
                        type="concept",
                        module_id="m1",
                    ),
                ],
            )
        ],
    )
    if with_state:
        progress.repetition_states["kp-series-resonance"] = RepetitionState(
            next_review_at=CORE_DUE,
            review_count=4,
            lapse_count=2,
            stability=5.0,
            last_review_at=NOW - 86400.0,
            desired_retention=0.9,
        )
    return progress


def test_linked_concept_reports_core_state(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress()
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)

    states = core_state.states_for_spec(linked_spec, service.store, now=NOW)

    assert states["c-series-resonance"].source == "core"
    assert states["c-series-resonance"].state.next_review_at == CORE_DUE
    assert states["c-quality-factor"].source == "core"


def test_unlinked_concept_keeps_local_state(
    service: ReviewService,
    fixture_spec: ReviewSpec,
) -> None:
    states = core_state.states_for_spec(fixture_spec, service.store, now=NOW)
    assert {state.source for state in states.values()} == {"review"}


def test_states_for_spec_reads_the_event_log_in_one_query(
    service: ReviewService,
    fixture_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.put_spec(fixture_spec)
    calls: list[tuple[object, ...]] = []
    original = storage_module.sqlite3.connect

    def counted(*args: object, **kwargs: object):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(storage_module.sqlite3, "connect", counted)
    core_state.states_for_spec(fixture_spec, service.store, now=NOW)
    # One connection for the pack's overlays and one for the entire event log.
    assert len(calls) == 2


def test_core_state_resolves_overlay_and_spec_paths(service, linked_spec, monkeypatch) -> None:
    service.put_spec(linked_spec)
    overlay_path = "overlay-path"
    overlay = _progress().model_copy(update={"book_id": overlay_path})
    native = _progress(with_state=False)
    seen = []

    def load(path_id):
        seen.append(path_id)
        return {overlay_path: overlay, linked_spec.mastery_path_id: native}[path_id]

    monkeypatch.setattr(core_state, "load_core_progress", load)
    service.store.change_link(linked_spec, linked_spec.concepts[0].id, (overlay_path, "kp-series-resonance"))
    states = core_state.states_for_spec(linked_spec, service.store, now=NOW)
    assert seen == [overlay_path, linked_spec.mastery_path_id]
    assert states[linked_spec.concepts[0].id].state.next_review_at == CORE_DUE
    assert states[linked_spec.concepts[1].id].initial
    assert service.store.get_spec(linked_spec.id).model_dump() == linked_spec.model_copy(
        update={"quality": service.store.get_spec(linked_spec.id).quality}
    ).model_dump()


def test_missing_core_path_falls_back_to_local(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: None)
    states = core_state.states_for_spec(linked_spec, service.store, now=NOW)
    assert {state.source for state in states.values()} == {"review"}


def test_linked_concept_without_core_state_gets_core_initial_state(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress(with_state=False)
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    states = core_state.states_for_spec(linked_spec, service.store, now=NOW)
    state = states["c-series-resonance"]
    assert state.source == "core"
    assert state.initial is True
    assert state.state.next_review_at >= NOW
    # The synthetic initial state is for display only.
    assert not progress.repetition_states


def test_core_known_not_due_is_not_mislabeled_new(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress()
    progress.repetition_states["kp-series-resonance"].next_review_at = NOW + 3600.0
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    service.put_spec(linked_spec)

    today = service.today(linked_spec.id, now=NOW)
    assert not [
        row for row in today["items"] if row["concept_id"] == "c-series-resonance"
    ]
    # The other linked concept has no core state yet, so it remains new.
    quality = [
        row for row in today["items"] if row["concept_id"] == "c-quality-factor"
    ]
    assert quality
    assert quality[0]["status"] == "new"
    assert quality[0]["state_source"] == "core"


def test_export_snapshot_uses_core_state(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress()
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    service.put_spec(linked_spec)

    html = service.export(linked_spec.id, include_snapshot=True, now=NOW)
    assert str(CORE_DUE) in html
    assert '"review_count":4' in html or '"review_count": 4' in html


def test_linked_concept_with_local_events_and_no_core_state_falls_back(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed projection must not hide a card the learner already answered."""
    progress = _progress(with_state=False)
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    service.put_spec(linked_spec)

    today = service.today(linked_spec.id, now=NOW)
    service.attempt(
        linked_spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="again",
            session_id=today["session_id"],
            submission_id="local-fallback-1",
            answer=3,
        ),
        now=NOW,
    )

    later = service.today(linked_spec.id, now=NOW + 0.5 * 86400.0)
    row = next(
        item
        for item in later["items"]
        if item["item_id"] == "item-resonance-choice"
    )
    assert row["state_source"] == "review"
    assert row["state_reason"] == "core_state_missing_local_history"
    assert row["status"] in {"due", "repeat"}


def test_legacy_core_state_is_hydrated_before_retrievability(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress()
    progress.repetition_states["kp-series-resonance"] = RepetitionState(
        next_review_at=NOW + 86400.0,
        interval_index=3,
        stability=0.0,
        last_review_at=NOW - 7 * 86400.0,
    )
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    service.put_spec(linked_spec)

    today = service.today(linked_spec.id, now=NOW + 2 * 86400.0)
    row = next(
        item for item in today["items"] if item["concept_id"] == "c-series-resonance"
    )
    assert row["state_source"] == "core"
    assert row["predicted_retrievability"] > 0.5


def test_mastery_path_allowlist_fails_closed(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress()
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    monkeypatch.setenv(core_state.ALLOWED_PATHS_ENV, "some-other-path")
    # The patched loader bypasses the env; assert the gate itself and its use
    # through the real loader separately.
    assert core_state.linkage_allowed("fixture-mastery") is False
    monkeypatch.setattr(
        core_state,
        "load_core_progress",
        lambda path: progress if core_state.linkage_allowed(path) else None,
    )
    service.put_spec(linked_spec)

    today = service.today(linked_spec.id, now=NOW)
    assert {row["state_source"] for row in today["items"]} == {"review"}
    assert core_state.linkage_allowed("fixture-mastery") is False


def test_mastery_path_allowlist_permits_named_path(
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(core_state.ALLOWED_PATHS_ENV, "fixture-mastery, other")
    assert core_state.linkage_allowed("fixture-mastery") is True
    assert core_state.linkage_allowed("other") is True
    assert core_state.linkage_allowed("third") is False


def test_today_and_attempt_use_core_state(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = _progress()
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    service.put_spec(linked_spec)

    today = service.today(linked_spec.id, now=NOW)
    item = next(
        row for row in today["items"] if row["item_id"] == "item-resonance-choice"
    )
    assert item["state_source"] == "core"
    assert item["due_at_ts"] == CORE_DUE

    result = service.attempt(
        linked_spec.id,
        AttemptInput(
            item_id="item-resonance-choice",
            rating="good",
            session_id=today["session_id"],
            submission_id="core-state-1",
            answer=1,
        ),
        now=NOW,
    )
    overlaid = service.apply_core_state(linked_spec.id, result, now=NOW)
    assert overlaid["state_source"] == "core"
    assert overlaid["next_due_ts"] == CORE_DUE
    assert overlaid["review_count"] == 4
    assert overlaid["lapse_count"] == 2
    # The local transition interval is overwritten so it cannot contradict the
    # core due date in the same response.
    assert overlaid["scheduled_interval_days"] == round(
        max(0.0, (CORE_DUE - result["event_ts"]) / 86400.0), 6
    )
    # The raw event values stay available for audit.
    assert overlaid["event_scheduled_interval_days"] == result[
        "event_scheduled_interval_days"
    ]
    assert overlaid["event_ts"] == result["event_ts"]


def _legacy_progress() -> LearningProgress:
    progress = _progress()
    progress.repetition_states["kp-series-resonance"] = RepetitionState(
        next_review_at=CORE_DUE,
        review_count=2,
        lapse_count=0,
        stability=0.0,
        interval_index=1,
        desired_retention=0.9,
    )
    return progress


def _linked_concept(spec: ReviewSpec):
    return next(c for c in spec.concepts if c.id == "c-series-resonance")


def test_core_scheduler_prefers_bound_strategy(
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StrategyScheduler:
        def hydrate(self, state, knowledge_type):
            state.stability = 42.0
            return state

        def get_initial_state(self, knowledge_type, *, now=None, desired_retention=None):
            return RepetitionState(next_review_at=NOW + 42.0, stability=42.0)

    progress = _legacy_progress()
    monkeypatch.setattr(
        core_state, "_strategy_scheduler", lambda value: _StrategyScheduler()
    )
    state, synthetic = core_state._core_state_for_indexed(
        linked_spec,
        _linked_concept(linked_spec),
        progress,
        core_state._kp_index(progress),
        now=NOW,
    )
    assert synthetic is False
    assert state is not None
    assert state.stability == 42.0


def test_core_scheduler_falls_back_when_strategy_resolution_fails(
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deeptutor.learning.scheduler import SpacedRepetitionScheduler

    def _boom(_progress):
        raise RuntimeError("bad binding")

    progress = _legacy_progress()
    monkeypatch.setattr(core_state, "_strategy_scheduler", _boom)
    state, _ = core_state._core_state_for_indexed(
        linked_spec,
        _linked_concept(linked_spec),
        progress,
        core_state._kp_index(progress),
        now=NOW,
    )
    expected = SpacedRepetitionScheduler(desired_retention=0.9).hydrate(
        progress.repetition_states["kp-series-resonance"].model_copy(deep=True),
        "concept",
    )
    assert state is not None
    assert state.stability == expected.stability


def test_core_scheduler_falls_back_without_review_strategy(
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deeptutor.learning.scheduler import SpacedRepetitionScheduler

    progress = _legacy_progress()
    monkeypatch.setattr(core_state, "_strategy_scheduler", None)
    state, _ = core_state._core_state_for_indexed(
        linked_spec,
        _linked_concept(linked_spec),
        progress,
        core_state._kp_index(progress),
        now=NOW,
    )
    expected = SpacedRepetitionScheduler(desired_retention=0.9).hydrate(
        progress.repetition_states["kp-series-resonance"].model_copy(deep=True),
        "concept",
    )
    assert state is not None
    assert state.stability == expected.stability


def test_states_for_spec_attaches_core_review_task(
    service: ReviewService,
    linked_spec: ReviewSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    progress = _progress()
    fake_task = SimpleNamespace(priority=2, forgetting_risk=0.42, reason="overdue")
    monkeypatch.setattr(core_state, "load_core_progress", lambda path: progress)
    monkeypatch.setattr(
        core_state,
        "_core_tasks",
        lambda value, *, now: {"kp-series-resonance": fake_task},
    )
    states = core_state.states_for_spec(linked_spec, service.store, now=NOW)
    assert states["c-series-resonance"].task is fake_task
    assert states["c-quality-factor"].task is None
