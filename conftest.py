"""
conftest.py — adds the datacleanenv project root to sys.path so that
`import models`, `import client`, and `from server.xxx import ...`
all resolve correctly when pytest is run from inside the datacleanenv/ directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Insert the project root (datacleanenv/) at the front of the module search path.
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
