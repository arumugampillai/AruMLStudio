"""PyInstaller runtime hook: fix a scipy.stats module-load crash under frozen builds.

Symptom
-------
The packaged EXE fails immediately at startup with:

    Failed to execute script 'master_dataset_manager' due to unhandled
    exception: name 'obj' is not defined

Root cause
----------
``scipy/stats/_distn_infrastructure.py`` ends with:

    for obj in [s for s in dir() if s.startswith('_doc_')]:
        exec('del ' + obj)
    del obj

This is a harmless docstring-cleanup loop under a normal interpreter. Under
PyInstaller's frozen importer (reproduced with Python 3.12.0; see
pyinstaller/pyinstaller#7992 and the CPython fix shipped in 3.12.1) the list
comprehension evaluates to an empty list, so the loop body never runs and
``obj`` is never bound. The trailing ``del obj`` then raises
``NameError: name 'obj' is not defined`` while the module is being imported,
which kills the whole app on startup. This app pulls scipy.stats in
transitively via ``sklearn.metrics`` (imported by
angelone/chart/chain_replay_ml/training/evaluator.py, which is reached from
master_dataset_tk/create_dataset_panel.py -> build_service.py ->
chain_replay_ml.dataset_builder.dataset_pipeline ->
chain_replay_ml.training.orchestrator -> .evaluator).

Fix
---
Install a meta path finder that intercepts the import of
``scipy.stats._distn_infrastructure``, reads its real source (this requires
the matching collection hook ``hook-scipy.stats._distn_infrastructure.py``
to set ``module_collection_mode = "pyz+py"`` so the .py file is actually
bundled, not just its bytecode), rewrites the unsafe ``del obj`` into an
equivalent that tolerates ``obj`` being unbound, and compiles/execs the
patched source in place of the original. The module behaves identically
afterwards — the names being deleted are only ever transient docstring-
building locals, never referenced again after this point.

The real, upstream fix is to build with Python 3.12.1+ (or 3.11.x), where
CPython's bug that breaks PyInstaller's frozen-importer code paths is fixed.
This runtime hook is a build-environment-independent safety net so the app
keeps working regardless of which Python 3.12.0.x patch level builds it.
"""

from __future__ import annotations

import sys

_TARGET_MODULE = "scipy.stats._distn_infrastructure"


def _patch_source(source: str) -> str:
    """Replace the unsafe trailing ``del obj`` with an unbound-safe equivalent."""
    target = "\ndel obj\n"
    replacement = "\nglobals().pop('obj', None)\n"
    if target in source:
        return source.replace(target, replacement, 1)
    return source


class _PatchedLoader:
    """Delegating loader that source-patches the target module before exec."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module) -> None:
        source = None
        try:
            source = self._inner.get_source(module.__name__)
        except Exception:
            source = None

        if not source:
            # No .py source bundled (collection hook missing/ineffective) —
            # best effort: pre-bind a sentinel so `del obj` has something to
            # remove even if the cleanup loop body never runs.
            module.__dict__.setdefault("obj", None)
            self._inner.exec_module(module)
            return

        patched = _patch_source(source)
        filename = getattr(self._inner, "path", module.__name__)
        exec(compile(patched, filename, "exec"), module.__dict__)


class _ScipyDistnInfrastructureFinder:
    """Meta path finder that wraps the loader for the target module only."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            try:
                real_spec = find(fullname, path, target)
            except Exception:
                continue
            if real_spec is None or real_spec.loader is None:
                continue
            real_spec.loader = _PatchedLoader(real_spec.loader)
            return real_spec
        return None


sys.meta_path.insert(0, _ScipyDistnInfrastructureFinder())
