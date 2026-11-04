"""The toolkit-independent refinement runner and structure previews, driven against the real backend.

A front-end drains the runner from its own GUI timer; here a plain loop stands in for that timer.

Not collected directly (no test_ prefix): test_gui_core_runner.py runs this file in a child process, because building a rigid body
permanently changes process-wide backend state (it sets settings::internal_state::prefer_partial_manager, which no settings API exposes or
resets), and that would break later tests in the same process that use atoms without form factors."""

import os
import time

import pytest

from pyausaxs.gui.core import preview, runner
from pyausaxs.gui.core.loads import RelaxedLoads

FILES = os.path.join(os.path.dirname(__file__), "files")
PDB, DAT = os.path.join(FILES, "2epe.pdb"), os.path.join(FILES, "2epe.dat")


def make_script(output_dir, *, splits="50 100", iterations=2, update=False) -> str:
    return f"""output {output_dir}/
load {{
    pdb {PDB}
    saxs {DAT}
    split {splits}
}}
parameters {{
    iterations {iterations}
    translate 1
    rotate 1
}}
loop
    optimize_once
        on_improvement
            {"update structure" if update else ""}
        end
    end
end
"""


def drain_until_done(worker, timeout=120.0):
    lines, outcomes = [], []
    deadline = time.monotonic() + timeout
    while worker.drain(lines.append, outcomes.append):
        assert time.monotonic() < deadline, "the run did not finish in time"
        time.sleep(0.02)
    assert len(outcomes) == 1
    return lines, outcomes[0]


class RecordingLoads(RelaxedLoads):
    """Records whether a grant is in force, to check the worker keeps it applied for exactly the duration of the run."""
    def __init__(self):
        super().__init__()
        self.applied_paths, self.restored = [], 0

    def apply(self, path):
        self.applied_paths.append(path)
        return {"marker": "1"}

    def restore(self, previous):
        if previous:
            self.restored += 1


def test_validation_succeeds(tmp_path):
    worker = runner.RefinementWorker()
    worker.start(make_script(tmp_path), validate_only=True)
    assert worker.running()
    _, outcome = drain_until_done(worker)
    assert outcome.error is None
    assert outcome.result is None
    assert not worker.running()


def test_validation_reports_a_bad_script(tmp_path):
    worker = runner.RefinementWorker()
    worker.start(make_script(tmp_path) + "no_such_element 1\n", validate_only=True)
    _, outcome = drain_until_done(worker)
    assert outcome.error is not None


def test_refinement_returns_fit_curves(tmp_path):
    worker = runner.RefinementWorker()
    worker.start(make_script(tmp_path), validate_only=False)
    lines, outcome = drain_until_done(worker)
    assert outcome.error is None, "".join(lines)
    assert outcome.result.ndim == 2 and outcome.result.shape[1] == 4
    assert lines, "the backend's console output should be streamed"


def test_a_second_start_while_running_is_refused(tmp_path):
    worker = runner.RefinementWorker()
    worker.start(make_script(tmp_path), validate_only=True)
    with pytest.raises(RuntimeError):
        worker.start(make_script(tmp_path), validate_only=True)
    drain_until_done(worker)


def test_relaxed_loads_are_held_until_the_run_is_drained(tmp_path):
    loads = RecordingLoads()
    worker = runner.RefinementWorker()
    worker.start(make_script(tmp_path), validate_only=True, structure_path=PDB, relaxed_loads=loads)
    assert loads.applied_paths == [PDB]
    assert loads.restored == 0
    drain_until_done(worker)
    assert loads.restored == 1


def test_stop_request_needs_a_run():
    assert not runner.RefinementWorker().request_stop()


# ── previews ───────────────────────────────────────────────────────────────
def test_preview_cache_rebuilds_only_on_structural_edits(tmp_path):
    cache = preview.PreviewCache()
    script = make_script(tmp_path)
    first = cache.get(script)
    assert first is not None and cache.error is None
    assert set(first["body"].tolist()) == {0, 1, 2}
    assert cache.get(script.replace("iterations 2", "iterations 7")) is first
    resplit = cache.get(make_script(tmp_path, splits="50"))
    assert resplit is not first
    assert set(resplit["body"].tolist()) == {0, 1}


def test_preview_cache_keeps_the_reason_for_a_failed_build(tmp_path):
    cache = preview.PreviewCache()
    assert cache.get(make_script(tmp_path, splits="1000000")) is None
    assert cache.error is not None
    cache.invalidate()
    assert cache.get(make_script(tmp_path)) is not None
    assert cache.error is None


def test_preview_cache_without_a_structure_is_empty():
    cache = preview.PreviewCache()
    assert cache.get("load {\n    pdb\n}\n") is None
    assert cache.error is None


def test_bare_structure_preview():
    data = preview.bare_structure_preview(PDB, DAT)
    assert data is not None and set(data["body"].tolist()) == {0}
    assert preview.bare_structure_preview(PDB, None) is None


def test_live_structure_follows_a_run(tmp_path):
    preview.LiveStructure.register_consumer()
    live = preview.LiveStructure()
    assert live.begin(make_script(tmp_path)) is None
    assert not live.active  # no `update` element: nothing to follow

    script = make_script(tmp_path, iterations=30, update=True)
    assert live.begin(script) is None
    assert live.active
    worker = runner.RefinementWorker()
    worker.start(script, validate_only=False)
    drain_until_done(worker)
    frame = live.final()
    assert not live.active
    # only improvements are published; 30 steps from an unrefined start reliably yield several (typically ~10)
    assert frame is not None
    assert frame["coords"].shape == (len(frame["body"]), 3)
