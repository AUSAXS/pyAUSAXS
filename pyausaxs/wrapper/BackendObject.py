from .AUSAXS import AUSAXS

class BackendObject:
    __slots__ = ['_object_id']

    def __init__(self):
        self._object_id: int = -1

    def __del__(self):
        ausaxs = AUSAXS()
        ausaxs.deallocate(self._object_id)