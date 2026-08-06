"""PyInstaller collection hook for ``catboost``.

catboost (a declared dependency in requirements.txt, imported lazily by
``chain_replay_ml/training/catboost_trainer.py``) ships a compiled
extension (``_catboost*.pyd``/``.so``) plus non-Python resource files
(e.g. HTML/JS assets used by its training-progress visualizer widget)
as package data. ``collect_all`` is the safe, catch-all way to make sure
none of that goes missing in a frozen build — mirroring the same fix
applied to xgboost in ``hook-xgboost.py`` for the same class of bug
("native library / package data not found in frozen EXE").

This hook is inert on machines where catboost is not installed: it is
only invoked by PyInstaller if catboost is actually reachable from the
app's import graph.
"""

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all("catboost")
