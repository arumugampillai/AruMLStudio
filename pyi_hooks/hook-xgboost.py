"""PyInstaller collection hook for ``xgboost``.

Symptom
-------
The packaged EXE crashes the moment xgboost is used (training or
inference) with:

    xgboost.core.XGBoostError: Cannot find XGBoost Library in the
    candidate path. List of candidates:
    ...\\_MEI...\\xgboost\\lib\\xgboost.dll
    ...\\_MEI...\\lib\\xgboost.dll
    ...

Root cause
----------
xgboost's compiled native library (``xgboost/lib/xgboost.dll`` on
Windows, ``libxgboost.so`` on Linux, ``libxgboost.dylib`` on macOS) is
shipped as package *data*, not as a normal importable extension module
(see ``xgboost/libpath.py::find_lib_path``, which looks for it at a
path relative to ``xgboost/__init__.py``). PyInstaller's static analysis
discovers binary dependencies by following ``import`` statements and
C-extension linkage — it has no way to know this loose .dll file needs
to be bundled at all, so a plain ``pyinstaller entry.py`` never ships
it, and the frozen app fails at first xgboost use.

Fix
---
``collect_all`` pulls in the compiled library (as a binary, so it also
picks up any transitive DLL dependencies PyInstaller can detect) plus
the package's other data files (e.g. ``xgboost/VERSION``, used by
``xgboost.core`` for version reporting) and hidden imports. Because this
file lives under ``pyi_hooks/`` and ``build.py`` already points
PyInstaller's ``--additional-hooks-dir`` there, it is picked up
automatically for any build that imports xgboost — no extra CLI flags
are required.

This mirrors the standard workaround for this well-known xgboost +
PyInstaller packaging gap (the same fix as passing ``--collect-all
xgboost`` on the command line).
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("xgboost")
