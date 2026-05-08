import pyausaxs as ausaxs
from pyausaxs.wrapper.Models import ExvModel, ExvTable


def test_settings():
    # call every setting with a concrete value to verify each setting name is
    # recognised by the AUSAXS backend
    ausaxs.settings.general(
        offline=True,
        verbose=True,
        warnings=True,
        threads=1,
    )
    ausaxs.settings.fit(
        fit_hydration=True,
        fit_excluded_volume=True,
        fit_solvent_density=True,
    )
    ausaxs.settings.grid(
        cell_width=1.0,
        expansion_factor=0.0,
        min_exv_radius=1.0,
    )
    ausaxs.settings.histogram(
        qmin=0.01,
        qmax=0.5,
        weighted_bins=True,
        bin_width=0.01,
        bin_count=100,
    )
    ausaxs.settings.molecule(
        throw_on_unknown_atom=False,
        implicit_hydrogens=True,
        use_occupancy=True,
        exv_table=ExvTable.traube,
    )
    ausaxs.settings.exv(exv_model=ExvModel.simple)


def test_settings_persistence():
    # Set two independent settings, then verify each holds its value.
    ausaxs.settings.general(verbose=True, threads=4)
    assert ausaxs.settings.get("verbose") == True
    assert ausaxs.settings.get("threads") == 4

    # Calling the setter with only one argument must not touch the other.
    ausaxs.settings.general(threads=8)
    assert ausaxs.settings.get("verbose") == True   # unchanged
    assert ausaxs.settings.get("threads") == 8      # updated

    # Explicitly updating a setting must overwrite the previous value.
    ausaxs.settings.general(verbose=False)
    assert ausaxs.settings.get("verbose") == False  # updated
    assert ausaxs.settings.get("threads") == 8      # unchanged

    # Cross-group: a setter from a different group must not affect these values.
    ausaxs.settings.histogram(qmin=0.05)
    assert ausaxs.settings.get("verbose") == False
    assert ausaxs.settings.get("threads") == 8
