import multiprocessing
import ctypes as ct
from enum import Enum

from pyausaxs.loader import find_lib_path
from pyausaxs.architecture import CPUFeatures

class AUSAXSLIB: 
    class STATE(Enum):
        UNINITIALIZED = 0
        FAILED = 1
        READY = 2

    def __init__(self):
        self.functions = None
        self.state = self.STATE.UNINITIALIZED
        self.lib_path = find_lib_path()

        self._check_cpu_compatibility()
        self._attach_hooks()
        self._test_integration()

    def _check_cpu_compatibility(self):
        """Check if the current CPU is compatible with the AUSAXS library."""
        if not CPUFeatures.is_compatible_architecture():
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Incompatible CPU architecture: {CPUFeatures.get_architecture()}")
        return True

    def _attach_hooks(self):
        # skip if CPU compatibility check already failed
        if self.state == self.STATE.FAILED:
            return

        self.state = self.STATE.READY
        try:
            self.functions = ct.CDLL(str(self.lib_path))

            # test_integration
            self.functions.test_integration.argtypes = [
                ct.POINTER(ct.c_int)    # test val
            ]
            self.functions.test_integration.restype = None

            # deallocate
            self.functions.deallocate.argtypes = [
                ct.c_int,               # object id
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.deallocate.restype = None

            # get_last_error_msg
            self.functions.get_last_error_msg.argtypes = [
                ct.POINTER(ct.c_char_p),    # msg (output)
                ct.POINTER(ct.c_int)        # status (0 = success)
            ]
            self.functions.get_last_error_msg.restype = None

            # read_pdb
            self.functions.pdb_read.argtypes = [
                ct.c_char_p,                         # filename
                ct.POINTER(ct.c_int)                 # status (0 = success)
            ]
            self.functions.pdb_read.restype = ct.c_int # return pdb id

            # pdb_get_data
            self.functions.pdb_get_data.argtypes = [
                ct.c_int,                            # object id
                ct.POINTER(ct.POINTER(ct.c_int)),    # serial (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # name (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # altLoc (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # resName (output)
                ct.POINTER(ct.POINTER(ct.c_char)),   # chainID (output)
                ct.POINTER(ct.POINTER(ct.c_int)),    # resSeq (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # iCode (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # x (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # y (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # z (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # occupancy (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # tempFactor (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # element (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # charge (output)
                ct.POINTER(ct.c_int),                # n_atoms (output)
                ct.POINTER(ct.c_int)                 # status (0 = success)
            ]
            self.functions.pdb_get_data.restype = ct.c_int # return data id

            # data_read
            self.functions.data_read.argtypes = [
                ct.c_char_p,            # filename
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.data_read.restype = ct.c_int # return data id

            # data_get_data
            self.functions.data_get_data.argtypes = [
                ct.c_int,                            # object id
                ct.POINTER(ct.POINTER(ct.c_double)), # q vector (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # I vector (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # Ierr vector (output)
                ct.POINTER(ct.c_int),                # n_points (output)
                ct.POINTER(ct.c_int)                 # status (0 = success)
            ]
            self.functions.data_get_data.restype = ct.c_int # return data id

            # molecule_from_file
            self.functions.molecule_from_file.argtypes = [
                ct.c_char_p,            # filename
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.molecule_from_file.restype = ct.c_int # return mol id

            # molecule_from_pdb_id
            self.functions.molecule_from_pdb_id.argtypes = [
                ct.c_int,               # pdb id
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.molecule_from_pdb_id.restype = ct.c_int # return mol id

            # molecule_from_arrays
            self.functions.molecule_from_arrays.argtypes = [
                ct.POINTER(ct.c_double), # x vector
                ct.POINTER(ct.c_double), # y vector
                ct.POINTER(ct.c_double), # z vector
                ct.POINTER(ct.c_double), # weight vector
                ct.c_int,                # n_atoms
                ct.POINTER(ct.c_int)     # status (0 = success)
            ]
            self.functions.molecule_from_arrays.restype = ct.c_int # return mol id

            # molecule_get_data
            self.functions.molecule_get_data.argtypes = [
                ct.c_int,                            # molecule id
                ct.POINTER(ct.POINTER(ct.c_double)), # ax_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # ay_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # az_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # aw_out (output)
                ct.POINTER(ct.POINTER(ct.c_char_p)), # aform_factors_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # wx_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # wy_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # wz_out (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # ww_out (output)
                ct.POINTER(ct.c_int),                # na (output)
                ct.POINTER(ct.c_int),                # nw (output)
                ct.POINTER(ct.c_int)                 # status (0 = success)
            ]
            self.functions.molecule_get_data.restype = ct.c_int # return data id

            # molecule_hydrate
            self.functions.molecule_hydrate.argtypes = [
                ct.c_int,               # molecule id
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.molecule_hydrate.restype = None

            # molecule_clear_hydration
            self.functions.molecule_clear_hydration.argtypes = [
                ct.c_int,               # molecule id
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.molecule_clear_hydration.restype = None

            # molecule_Rg
            self.functions.molecule_Rg.argtypes = [
                ct.c_int,               # molecule id
                ct.POINTER(ct.c_double),# Rg (output)
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.molecule_Rg.restype = None

            # molecule_distance_histogram
            self.functions.molecule_distance_histogram.argtypes = [
                ct.c_int,                            # molecule id
                ct.POINTER(ct.POINTER(ct.c_double)), # aa (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # aw (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # ww (output)
                ct.POINTER(ct.c_int),                # n_bins (output)
                ct.POINTER(ct.c_double),             # delta_r (output)
                ct.POINTER(ct.c_int)                 # status (0 = success)
            ]
            self.functions.molecule_distance_histogram.restype = ct.c_int # return obj id

            # molecule_debye
            self.functions.molecule_debye.argtypes = [
                ct.c_int,                            # molecule id
                ct.POINTER(ct.POINTER(ct.c_double)), # q (output)
                ct.POINTER(ct.POINTER(ct.c_double)), # I (output)
                ct.POINTER(ct.c_int),                # n_points (output)
                ct.POINTER(ct.c_int)                 # status (0 = success)
            ]
            self.functions.molecule_debye.restype = ct.c_int # return obj id

            # molecule_debye_userq
            self.functions.molecule_debye_userq.argtypes = [
                ct.c_int,                # molecule id
                ct.POINTER(ct.c_double), # q
                ct.POINTER(ct.c_double), # I (output)
                ct.c_int,                # n_points
                ct.POINTER(ct.c_int)     # status (0 = success)
            ]
            self.functions.molecule_debye_userq.restype = None

            # molecule_debye_fit
            self.functions.molecule_debye_fit.argtypes = [
                ct.c_int,              # molecule id
                ct.c_int,              # data id
                ct.POINTER(ct.c_int)   # status (0 = success)
            ]
            self.functions.molecule_debye_fit.restype = int # return res id

            # pdb_debye_fit
            self.functions.pdb_debye_fit.argtypes = [
                ct.c_int,              # pdb id
                ct.c_int,              # data id
                ct.POINTER(ct.c_int)   # status (0 = success)
            ]
            self.functions.pdb_debye_fit.restype = None # return res id

            # fit_get_fit_info
            self.functions.fit_get_fit_info.argtypes = [
                ct.c_int,                               # fit id
                ct.POINTER(ct.POINTER(ct.c_char_p)),    # pars (output)
                ct.POINTER(ct.POINTER(ct.c_double)),    # pvals (output)
                ct.POINTER(ct.POINTER(ct.c_double)),    # perr_min (output)
                ct.POINTER(ct.POINTER(ct.c_double)),    # perr_max (output)
                ct.POINTER(ct.c_int),                   # n_pars (output)
                ct.POINTER(ct.c_double),                # chi_squared (output)
                ct.POINTER(ct.c_int),                   # dof (output)
                ct.POINTER(ct.c_int)                    # status (0 = success)
            ]
            self.functions.fit_get_fit_info.restype = ct.c_int # return data id

            # fit_get_fit_curves
            self.functions.fit_get_fit_curves.argtypes = [
                ct.c_int,                               # fit id
                ct.POINTER(ct.POINTER(ct.c_double)),    # q (output)
                ct.POINTER(ct.POINTER(ct.c_double)),    # I_data (output)
                ct.POINTER(ct.POINTER(ct.c_double)),    # I_err (output)
                ct.POINTER(ct.POINTER(ct.c_double)),    # I_model (output)
                ct.POINTER(ct.c_int),                   # n_points (output)
                ct.POINTER(ct.c_int)                    # status (0 = success)
            ]
            self.functions.fit_get_fit_curves.restype = ct.c_int # return data id

            # debye_no_ff
            self.functions.debye_no_ff.argtypes = [
                ct.POINTER(ct.c_double), # q vector
                ct.POINTER(ct.c_double), # atom x vector
                ct.POINTER(ct.c_double), # atom y vector
                ct.POINTER(ct.c_double), # atom z vector
                ct.POINTER(ct.c_double), # atom weight vector
                ct.c_int,                # nq (number of points in q)
                ct.c_int,                # nc (number of points in x, y, z, w)
                ct.POINTER(ct.c_double), # Iq vector for return value
                ct.POINTER(ct.c_int)     # status (0 = success)
            ]
            self.functions.debye_no_ff.restype = None

            # set_exv_settings
            self.functions.set_exv_settings.argtypes = [
                ct.c_char_p,            # exv_model
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.set_exv_settings.restype = None

            # set_fit_settings
            self.functions.set_fit_settings.argtypes = [
                ct.c_uint,              # sampled points
                ct.c_uint,              # max_iterations
                ct.c_bool,              # fit_excluded_volume
                ct.c_bool,              # fit_solvent_density
                ct.c_bool,              # fit_hydration
                ct.c_bool,              # fit_atomic_debye_waller
                ct.c_bool,              # fit_exv_debye_waller
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.set_fit_settings.restype = None

            # set_grid_settings
            self.functions.set_grid_settings.argtypes = [
                ct.c_double,            # water_scaling
                ct.c_double,            # cell_width
                ct.c_double,            # scaling
                ct.c_double,            # min_exv_radius
                ct.c_uint,              # min_bins
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.set_grid_settings.restype = None

            # set_hist_settings
            self.functions.set_hist_settings.argtypes = [
                ct.c_uint,              # skip
                ct.c_double,            # qmin
                ct.c_double,            # qmax
                ct.c_bool,              # weighted_bins
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.set_hist_settings.restype = None

            # set_molecule_settings
            self.functions.set_molecule_settings.argtypes = [
                ct.c_bool,              # center
                ct.c_bool,              # throw_on_unknown_atom
                ct.c_bool,              # implicit_hydrogens
                ct.c_bool,              # use_occupancy
                ct.c_char_p,            # exv_set
                ct.c_char_p,            # hydration_strategy
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.set_molecule_settings.restype = None

            # set_general_settings
            self.functions.set_general_settings.argtypes = [
                ct.c_bool,              # offline
                ct.c_bool,              # verbose
                ct.c_bool,              # warnings
                ct.c_uint,              # threads
                ct.POINTER(ct.c_int)    # status (0 = success)
            ]
            self.functions.set_general_settings.restype = None

            # iterative_fit_start
            self.functions.iterative_fit_start.argtypes = [
                ct.c_int,               # molecule id
                ct.c_int,               # data id
                ct.POINTER(ct.c_int)    # return status (0 = success)
            ]
            self.functions.iterative_fit_start.restype = ct.c_int # return iterative fit id

            # iterative_fit_step
            self.functions.iterative_fit_step.argtypes = [
                ct.c_int,               # iterative fit id
                ct.POINTER(ct.c_double),# parameters vector
                ct.c_int,               # number of parameters
                ct.POINTER(ct.c_double),# return I vector for return value
                ct.POINTER(ct.c_int),   # return status (0 = success)
            ]
            self.functions.iterative_fit_step.restype = None

            self.state = self.STATE.READY

        except Exception as e:
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Unexpected error during library integration: {e}")

    def _test_integration(self):
        """
        Test the integration of the AUSAXS library by running a simple test function in a separate process. 
        This protects the main thread from potential segfaults due to e.g. incompatible architectures. 
        """
        if (self.state != self.STATE.READY):
            return

        try: 
            # we need a queue to access the return value
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=_run, args=(self.lib_path, queue))
            p.start()
            p.join()
            if p.exitcode == 0: # process successfully terminated
                val = queue.get_nowait() # get the return value
                if (val != 6): # test_integration increments the test value by 1
                    raise Exception("AUSAXS integration test failed. Test value was not incremented")
            else:
                raise Exception(f"AUSAXS: External invocation seems to have crashed (exit code \"{p.exitcode}\").")

        except Exception as e:
            self.state = self.STATE.FAILED
            raise RuntimeError(f"AUSAXS: Unexpected integration test failure: \"{e}\".")

    def ready(self):
        return self.state == self.STATE.READY

def _run(lib_path, queue):
    """
    Helper method for AUSAXSLIB._test_integration, which must be defined in global scope to be picklable.
    """
    func = ct.CDLL(str(lib_path))
    func.test_integration.argtypes = [ct.POINTER(ct.c_int)]
    func.test_integration.restype = None
    test_val = ct.c_int(5)
    func.test_integration(ct.byref(test_val))
    queue.put(test_val.value)