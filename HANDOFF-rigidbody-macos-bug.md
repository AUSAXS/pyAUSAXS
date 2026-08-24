# Handoff: rigid-body refinement fails at the very end, GUI-only, macOS-only

Status as of 2026-08-24. Written for whoever picks this up next. Nothing is committed in
`pyausaxs`; the `ausaxs` changes described below **were** committed by Kristian.

---

## 1. The symptom

A user (Katharina, macOS/Apple Silicon) runs a rigid-body refinement in the **GUI**. The refinement
runs to completion — every iteration prints, `final_state.pdb` and `trajectory.xyz` are written to
disk — and then the Output pane prints, in red:

```
Output written to file /Users/katharina/output/rigidbody/final_state.pdb.
Trajectory written to /Users/katharina/output/rigidbody/trajectory.xyz

Refinement failed.
```

The original screenshot is `Screenshot 2026-08-24 at 15.28.22.png` in the repo root.

### Established facts (each one cost real effort — do not re-derive)

| Fact | How we know |
|---|---|
| The **CLI works**. `ausaxs rigidbody <same script>` completes fine. | She ran it. |
| It is **not data-dependent** — *every* setup from Kristian's tutorial fails. | He tested. |
| It is **not the live-structure polling**. Removing `update structure` still fails. | She tested. |
| It is **not an abort/assert** — she runs a **debug build with asserts on**, and the GUI survives. | Kristian. |
| Nothing appears on **stderr**, and she launches the GUI **from a terminal**. | Kristian. |
| Her pyausaxs is **v1.2.1 from PyPI** — exactly `HEAD` of this repo. | Kristian. |
| Not reproducible on Linux. Tried: same script shape, CIF input, `symmetry b1 p2`, 3-column/79-point data, worker thread + output callback + live preview, against both the v1.2.9 release `.so` and a fresh local build. All pass. | See §5. |

## 2. Why "Refinement failed." pins down the failing statement

`RigidbodyPane._on_done` prints that exact string in **one** place, guarded by `done.error_streamed`.
`error_streamed` is set in **one** place: the inner `except` around `rb.run()` in
`RigidbodyRunner._work` (`pyausaxs/gui/runner.py`). Therefore **`Rigidbody.run()` raised a Python
exception.**

It is *not* the plotting: a plot failure prints `Failed to plot results: …` and is always preceded by
`Refinement completed.`, and she sees neither.

`Rigidbody.run()` (v1.2.1) has exactly five statements that can raise:

| statement | verdict |
|---|---|
| `AUSAXS()` | impossible — singleton already built |
| `ausaxs.lib()` | impossible — raises only if not ready, and the refinement demonstrably ran |
| `self._get_id()` | impossible — evaluated *before* the call; the run happened |
| the ctypes call `rigidbody_run(...)` | **possible** |
| `_check_error_code(status, …)` | **possible, but implies `status != 0`** |
| `np.column_stack((...))` | **ruled out** — see below |

**Why `np.column_stack` is ruled out.** For it to raise, one of the four returned pointers must be
NULL while `n_points > 0` (then `_ptr_to_array` returns a length-0 array for that one and length-n
for the others). But `Dataset::select_columns({0,1,2,3})` allocates its destination as
`Dataset(this->data.N, 4)` and fills all four columns, so the four `std::vector<double>`s always have
exactly N elements. Either N > 0 and no pointer is NULL, or N == 0 and all four are — and that second
case yields shape `(0, 4)`, which prints **"No fit curves were returned."**, a different message.

**The remaining contradiction.** `execute_with_catch` writes `*status = 1` on entry and `*status = 0`
on success, per call. So `status != 0` ⟺ a C++ exception was caught ⟺ `report_api_exception` wrote to
stderr. She reports no stderr. That is the knot that has not been untied.

## 3. IMPORTANT — a correction to earlier reasoning

Much of the analysis above leaned on "no red error line in her GUI console, therefore no ausaxs
exception". **That inference was invalid for her latest run.** The screenshot predates Kristian's
`use_ausaxs_exceptions` refactor. In the *new* backend, `except::base`'s constructor calls
`console::print_critical` → `std::cout` → the GUI console, so every deliberate ausaxs throw now
prints itself in red directly above "Refinement failed.".

**First thing to do: get the full console text from her NEW build, not the old screenshot.** If a red
line is there, it names the cause outright. If it genuinely is not, then every *deliberate* throw is
excluded (there are now zero `throw std::` in `source/`) and what remains is an **implicit** std throw
— `unordered_map::at`, `bad_optional_access`, `bad_alloc`, `std::filesystem`, `stod` — which prints
nothing to cout and lands only on stderr.

## 4. The macOS runner (how to reproduce remotely)

`.github/workflows/macos-tmate-debug.yml` (workflow_dispatch) spins up `macos-latest` (**arm64**),
checks out the repo, sets up Python 3.10, downloads the **latest AUSAXS release** dylib, and opens a
tmate session. Kristian intends to replace it with a proper SSH action — **prefer that**; tmate only
accepts a tmux attach, not `ssh host 'cmd'`.

Environment left ready on the runner (may be gone by now):

```
repo:    /Users/runner/work/pyAUSAXS/pyAUSAXS   (v1.2.1, commit 3acf2d3)
dylib:   lib/libausaxs.dylib  -> copied to pyausaxs/resources/libausaxs.dylib
pip:     numpy 2.2.6, py-cpuinfo 9.0.0   (both were missing; install them first)
scripts: /tmp/diag.py  (= scripts/diagnose_rigidbody.py)
         /tmp/rb.txt   (a refinement script over tests/files/2epe.{pdb,dat}, symmetry b1 p2)
run:     cd <repo> && PYTHONPATH=. python3 /tmp/diag.py /tmp/rb.txt
```

Note the runner's dylib is the **released v1.2.9**, which does *not* contain Kristian's exception
refactor or the ObjectStorage mutex. To test the build she actually has, download a newer artifact.

If you must use tmate again, the working recipe is in the scratchpad `tm.sh`: pipe commands into
`ssh -tt`, tag every invocation with a unique token, write output to `/tmp/o.$TOKEN`, then `cat` it
back and grep for the token — the pane replays stale scrollback otherwise, which will fool you.
**Never send `exit`**: it kills the shell and ends the debug session. Detach with `\002d` (Ctrl-B d).
`base64 -d` does not decode on macOS (BSD); use `base64 -D` or python.

## 4b. A real macOS-only hazard found on the runner (probably not the bug, but know about it)

`AUSAXSLIB._test_integration` (`pyausaxs/integration.py`) runs the backend self-test in a
`multiprocessing.Process`. **macOS defaults to the `spawn` start method**, which re-imports the
`__main__` module in the child. Any entry point without an `if __name__ == "__main__":` guard
therefore re-runs itself in the child and dies with:

```
RuntimeError: AUSAXS: library failed to initialize. Reason: AUSAXS: Unexpected integration test
failure: "An attempt has been made to start a new process before the current process has finished
its bootstrapping phase. ... freeze_support() ..."
```

This bit `scripts/diagnose_rigidbody.py` on the runner; it is now wrapped in `main()` with a guard
and a `freeze_support()` call. **It is not her bug** — her GUI initialises fine, so the self-test
passes there — but it is a genuine macOS-only fragility in the library-init path, and it will bite
any new script you write. Worth considering whether `_test_integration` should force the `fork`
start method or use a dedicated non-`__main__` entry point.

## 5. What was already tried and did NOT reproduce (on Linux)

- v1.2.9 release `.so` and a fresh local debug build, both fine.
- GUI-shaped path: worker thread + `set_output_callback` + `preview_structure()` + live polling.
- CIF input (`6LYZ.cif`), `symmetry b1 p2`, 3-column 79-point dataset — matching her setup.
- 8 threads × 3000 concurrent `live_structure()` polls during a run: no failure, no duplicate ids.
- A second run after the first freed its result.

## 6. Changes made (uncommitted, in this repo)

- **`gui/runner.py` + `gui/rigidbody_pane.py`** — the GUI was *discarding* the exception message.
  `error_streamed` has been removed; `_on_done` now prints `Refinement failed: <reason>`. It first
  checks `ConsolePane.tail()` (new, in `gui/widgets.py`) to avoid echoing a message `except::base`
  already streamed. **Ship this** — it is what makes her next run self-explanatory.
- **`wrapper/Rigidbody.py`** — `run()` now deallocates the id `rigidbody_run` returns (it was leaking
  the result vectors on every refinement), and validates the four column lengths so a short column is
  named instead of surfacing as an opaque numpy shape error.
- **`wrapper/Rigidbody.py` + `gui/*`** — new `LivePoller` class replacing `Rigidbody.live_structure()`
  (see §7), built lazily in `_begin_live_preview` so a backend without the symbols degrades to
  "Live structure view unavailable" rather than breaking the pane.
- **`scripts/diagnose_rigidbody.py`** — runs the GUI path in a terminal and prints the full traceback,
  the raw `status`/`n_points`/pointer values, the four column lengths, and a
  `>>> STDERR IS VISIBLE <<<` marker.
- `tests/test_rigidbody.py` — one added test. Suite: **36 passed, 2 skipped**.

## 7. Changes committed in `ausaxs` by Kristian (already in his tree)

- `bb256fe7 add mutex to objectstorage` — `api::ObjectStorage` was completely unsynchronised
  (`current_id` a plain `int`, `storage` a plain `unordered_map`) while the GUI hammered it from the
  Tk thread. Verified concretely: the old header **segfaults/aborts within seconds** under 8 threads;
  the mutexed version is clean (0 duplicate ids, 0 leaked entries). Also added
  `rigidbody_create_live_poller` / `rigidbody_poll_live_structure`, a persistent-buffer poller so a
  polling GUI never touches the object storage.
- `3915e1a9 all gui errors are now written to cerr` — `report_api_exception` in `execute_with_catch`,
  writing to **`std::cerr`** (not `std::cout`, which `set_output_callback` redirects into the GUI).

**This did not fix her bug.** It is still worth having.

## 8. Open leads, best first

1. **Get the new build's full console text.** See §3. Cheapest, highest information.
2. **Confirm stderr really is visible** — run `scripts/diagnose_rigidbody.py` and look for the
   `>>> STDERR IS VISIBLE <<<` marker. Everything in §2 hinges on this being true.
3. **Reproduce on the arm64 runner.** The environment in §4 was one command away from a result when
   this session ended.
4. **Latent bug worth fixing regardless** (not her symptom): 31 sites across `source/api/` do
   `ErrorMessage::last_error = "..."; return -1;` **without throwing**, so `execute_with_catch` then
   sets `*status = 0` — a failure reported to Python as *success*, with `last_error` left stale for
   the next reader. `rigidbody_run`'s own id guard is one
   (`source/api/pyausaxs/api_rigidbody.cpp:227`). Converting them to
   `throw except::invalid_argument(...)` is mechanical and matches where the refactor is heading.
5. `ErrorMessage::last_error` is a process-global written by any thread with no pairing to the failing
   call. If a message *does* come back, it may belong to a different call.

## 9. Dead ends — do not spend time here again

- **The "Trajectory written to …" line is not a symptom.** It is printed from `XYZWriter::~XYZWriter()`.
  Commit `2666692b` made `SaveElement::~SaveElement()` call `reset_statics()` → `writers.clear()`, so
  from **v1.2.9** onward it appears at the end of every run. Before that the writer was a
  function-local static destroyed only at process exit, so it went to the real stdout. It is a
  **version marker**, nothing more. (This is why Kristian did not see it locally: his
  `~/.cache/ausaxs/libpath` relinks to his own older build.)
- The extra constrained `get_fitter()->fit()` in `rigidbody_run` (which the CLI does not do) looked
  promising but the parameter-count-mismatch mechanism needs asserts to be off, and she runs a debug
  build. Ruled out by Kristian.
- `multiprocessing` spawn-vs-fork on macOS: `AUSAXSLIB._test_integration` does spawn a subprocess, but
  `reset_singleton` is never called by the GUI, so it runs once at startup and demonstrably succeeds.
- The ObjectStorage id race: real, fixed, and **not** her bug (she still fails, and with
  `update structure` removed there is no concurrent polling at all).
