"""Suite-wide setup.

The library persistence file (projects / track metadata / notes) is
pointed at a throwaway temp file so test runs never read or write the
developer's real library, and every session starts clean.
"""
import os
import tempfile

_LIBRARY_DIR = tempfile.mkdtemp(prefix="pythagora-test-library-")
os.environ["PYTHAGORAS_LIBRARY_FILE"] = os.path.join(_LIBRARY_DIR, "library.json")
