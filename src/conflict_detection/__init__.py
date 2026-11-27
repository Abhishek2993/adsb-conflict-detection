"""Short-term conflict detection and resolution on ADS-B trajectories.

Pipeline: ingest -> clean -> predict -> CPA -> detect -> resolve -> BlueSky.
"""

from .config import Config, SeparationMinima

__all__ = ["Config", "SeparationMinima", "__version__"]
__version__ = "0.1.0"
