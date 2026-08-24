#!/usr/bin/env python3
"""Reproduce the GUI's rigid-body path in a terminal, and print what it actually does.

    python3 diagnose_rigidbody.py <script.txt>

Runs the refinement exactly the way the GUI does -- in-process, on a worker thread, with the
backend's output routed through the GUI's callback -- but prints the full Python traceback and
the raw values that come back across the C boundary, instead of collapsing it to
"Refinement failed."
"""
import ctypes as ct
import multiprocessing
import sys, threading, traceback

def main():
    multiprocessing.freeze_support()
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    script = open(sys.argv[1], encoding="utf-8").read()

    import pyausaxs
    from pyausaxs.wrapper.AUSAXS import AUSAXS, _check_error_code, _ptr_to_array
    from pyausaxs.wrapper.Rigidbody import Rigidbody, prepare_rigidbody_refinement
    from pyausaxs.wrapper.Output import set_output_callback, reset_output_callback

    import numpy, platform
    print(f"pyausaxs {pyausaxs.__version__}   numpy {numpy.__version__}   python {platform.python_version()}")
    print(f"platform {platform.platform()}  {platform.machine()}")
    print(f"backend  {AUSAXS().lib().lib_path}")
    print(f"ready    {AUSAXS().ready()}  {AUSAXS().init_error() or ''}")

    # The backend writes exceptions to stderr. If this marker does not appear, stderr is not reaching
    # the terminal here either, and the absence of a backend error message proves nothing.
    print(sys.stderr, file=sys.stderr)
    print(">>> STDERR IS VISIBLE <<<", file=sys.stderr, flush=True)
    print()

    def run_like_the_gui():
        """rigidbody_run, unpacked step by step so a failure says which step failed."""
        ausaxs = AUSAXS()
        q, I = ct.POINTER(ct.c_double)(), ct.POINTER(ct.c_double)()
        I_err, I_interp = ct.POINTER(ct.c_double)(), ct.POINTER(ct.c_double)()
        n_points, status = ct.c_int(), ct.c_int()

        rb = prepare_rigidbody_refinement(script)
        data_id = ausaxs.lib().functions.rigidbody_run(
            rb._get_id(), ct.byref(q), ct.byref(I), ct.byref(I_err), ct.byref(I_interp),
            ct.byref(n_points), ct.byref(status))

        print("\n----- what came back across the C boundary -----")
        print(f"  status    = {status.value}   (0 = the backend reported success)")
        print(f"  data id   = {data_id}")
        print(f"  n_points  = {n_points.value}")
        for name, p in (("q", q), ("I", I), ("I_err", I_err), ("I_interp", I_interp)):
            addr = ct.cast(p, ct.c_void_p).value
            print(f"  {name:<9}= {'NULL' if not p else hex(addr)}")
        print("-----------------------------------------------\n")

        _check_error_code(status, "rigidbody_run")
        n = n_points.value
        cols = [_ptr_to_array(p, n) for p in (q, I, I_err, I_interp)]
        print("column lengths:", [len(c) for c in cols], f"(all should be {n})")
        import numpy as np
        return np.column_stack(cols)

    outcome = {}
    def worker():                      # the GUI runs the refinement on a worker thread, not the main one
        set_output_callback(sys.stdout.write)   # the GUI installs a callback; the CLI does not
        try:
            outcome["result"] = run_like_the_gui()
        except BaseException:
            outcome["traceback"] = traceback.format_exc()
        finally:
            reset_output_callback()

    t = threading.Thread(target=worker)
    t.start(); t.join()

    print("\n===============================================")
    if "traceback" in outcome:
        print("THE RUN FAILED. This is what the GUI hides behind \"Refinement failed.\":\n")
        print(outcome["traceback"])
        sys.exit(1)
    print(f"The run succeeded: result shape {outcome['result'].shape}")


if __name__ == "__main__":
    main()
