import pyausaxs as ausaxs


def test_settings():
    # just call all settings to ensure no errors occur
    ausaxs.settings.general()
    ausaxs.settings.fit()
    ausaxs.settings.grid()
    ausaxs.settings.histogram()
    ausaxs.settings.molecule()
    ausaxs.settings.exv()
