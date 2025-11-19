import pyausaxs as ausaxs


def test_settings():
    ausaxs.settings.general()
    ausaxs.settings.fit()
    ausaxs.settings.grid()
    ausaxs.settings.histogram()
    ausaxs.settings.molecule()
    ausaxs.settings.exv()
