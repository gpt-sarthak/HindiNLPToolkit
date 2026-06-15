"""
webapp
======
Web layer for the Hindi NLP Toolkit.  Imports from the logic packages
(stanza_parser, filtering, variants, scoring) — never the reverse.

Run from the project root with:

    python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Make the logic packages importable regardless of how the app is launched.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
