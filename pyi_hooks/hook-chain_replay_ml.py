"""PyInstaller collection hook for ``chain_replay_ml``.

Symptom
-------
The packaged EXE crashes as soon as the Create Dataset panel touches the
OHLC Aggregation transformation, with:

    FileNotFoundError: ...\\_MEI...\\chain_replay_ml\\dataset_builder\\
    transformations\\ohlc_history_profiles.json

Root cause
----------
Several ``chain_replay_ml`` modules ship a small JSON config file as a
sibling of their ``.py`` source and load it at runtime via
``Path(__file__).with_name(...)`` (e.g.
``dataset_builder/transformations/ohlc_history_profiles.py`` ->
``ohlc_history_profiles.json``, and
``dataset_builder/transformations/horizon_policy.py`` ->
``horizon_policy.json``). This is a deliberate "edit the JSON, no code
changes needed" design — see those modules' docstrings — and works fine
from a normal checkout because the .py and .json files live in the same
directory on disk.

PyInstaller's static analysis only follows Python ``import`` statements;
it has no way to know a module reads an arbitrary sibling file at
runtime, so a plain ``pyinstaller entry.py`` never bundles these JSON
files. When frozen, ``__file__`` for an extracted module points inside
the ``_MEIPASS`` temp dir, ``.with_name("...json")`` resolves to a path
that was simply never extracted there, and ``Path.read_text()`` raises
``FileNotFoundError``.

Fix
---
``collect_data_files`` bundles the ``*.json`` sidecars from
``dataset_builder/transformations/`` — the exact directory both modules'
``Path(__file__).with_name(...)`` calls resolve against — so any future
config JSON dropped in that same folder (the documented, no-code-change
way to extend these two modules) keeps working without touching this
hook again.

This is intentionally scoped to that one subdirectory rather than a
blanket ``collect_data_files("chain_replay_ml")`` / ``--collect-data
chain_replay_ml`` over the whole package: the package tree also contains
generated, machine-local, and potentially sensitive non-.py files that
must never ship in the EXE — e.g. ``data/replay_session.db`` (a live
SQLite database of the user's own replay sessions), ``data/replay_cache/``
artifacts, ``dataset_builder/scripts/*.json`` audit dumps,
``performance/**/__pycache__`` numba JIT cache files (``.nbi``/``.nbc``),
and ``regression/golden_*.json`` test fixtures. A blanket collect would
silently bundle all of that into every release.

Because this file lives under ``pyi_hooks/`` and ``build.py`` already
points PyInstaller's ``--additional-hooks-dir`` there, it is picked up
automatically for any build that imports ``chain_replay_ml`` — no extra
CLI flags are required.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files(
    "chain_replay_ml",
    subdir="dataset_builder/transformations",
    includes=["*.json"],
)
