from .AUSAXS import AUSAXS

class BackendObject:
    __slots__ = ['_object_id', '_auto_gc']

    def _get_id(self) -> int:
        """Get the underlying C++ object ID."""
        if self._object_id == -1:
            raise RuntimeError("BackendObject: Attempted to use an invalid or deallocated object.")
        return self._object_id
    
    def _set_id(self, id: int) -> None:
        """Set the underlying C++ object ID."""
        self._object_id = id

    def __init__(self):
        self._object_id: int = -1
        self._auto_gc = True

    def __del__(self):
        if self._auto_gc and self._object_id != -1:
            ausaxs = AUSAXS()
            ausaxs.deallocate(self._object_id)

class advanced:
    @staticmethod
    def detach(backend_object: BackendObject):
        """
        Detach the lifetime of the underlying C++ object from the BackendObject instance.
        After invoking  this method, _you_ are responsible for invoking `deallocate` on the object for proper cleanup.
        """
        backend_object._auto_gc = False
    
    @staticmethod
    def attach(backend_object: BackendObject):
        """
        Reattach the lifetime of the underlying C++ object to the BackendObject instance.
        After invoking  this method, the BackendObject instance will automatically deallocate the C++ object upon destruction.
        """
        backend_object._auto_gc = True
    
    @staticmethod
    def deallocate(backend_object: BackendObject):
        """
        Immediately deallocate the underlying C++ object associated with the BackendObject instance.
        After invoking this method, the object is rendered unusable. 
        """
        if backend_object._object_id != -1:
            ausaxs = AUSAXS()
            ausaxs.deallocate(backend_object._object_id)
            backend_object._object_id = -1
            backend_object._auto_gc = False