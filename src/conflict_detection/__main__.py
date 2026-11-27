"""Allow ``python -m conflict_detection``."""

import sys

from .cli import main

sys.exit(main())
