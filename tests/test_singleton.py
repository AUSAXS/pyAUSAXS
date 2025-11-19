import pyausaxs as ausaxs


def test_singleton():
    """Test that AUSAXS instances are the same object."""
    cls = ausaxs.wrapper.AUSAXS.AUSAXS
    instance1 = cls()
    instance2 = cls()
    instance3 = cls()
    assert instance1 is instance2, "Instance 1 and 2 should be the same object"
    assert instance2 is instance3, "Instance 2 and 3 should be the same object"
    assert instance1 is instance3, "Instance 1 and 3 should be the same object"
    assert instance1.ready() == instance2.ready(), "All instances should have the same ready state"
    assert instance1.init_error() == instance2.init_error(), "All instances should have the same error state"


def test_reset_singleton():
    """Test that reset_singleton works correctly."""
    cls = ausaxs.wrapper.AUSAXS.AUSAXS
    instance1 = cls()
    ready1 = instance1.ready()
    cls.reset_singleton()
    instance2 = cls()
    ready2 = instance2.ready()
    assert instance1 is not instance2, "After reset, new instance should be a different object"
    assert ready1 == ready2, "Ready state should be consistent across resets"
