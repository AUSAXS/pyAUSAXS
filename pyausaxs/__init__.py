"""pyausaxs package init"""

from .wrapper import AUSAXS, AUSAXSManualFit, ausaxs, create_ausaxs

__all__ = ["AUSAXS", "AUSAXSManualFit", "ausaxs", "create_ausaxs"]
__version__ = "1.1.0"
ausaxs = ausaxs