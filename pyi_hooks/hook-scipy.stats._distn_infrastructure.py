"""PyInstaller collection hook for ``scipy.stats._distn_infrastructure``.

That module ends with a module-level docstring cleanup loop:

    for obj in [s for s in dir() if s.startswith('_doc_')]:
        exec('del ' + obj)
    del obj

Under PyInstaller's frozen importer (reproduced here with Python 3.12.0 —
see https://github.com/pyinstaller/pyinstaller/issues/7992 and the CPython
fix in 3.12.1) the list comprehension evaluates to an empty list, so the
loop body never runs and ``obj`` is never bound. The trailing ``del obj``
then raises ``NameError: name 'obj' is not defined``, which crashes the
whole app the moment anything imports scipy.stats — this app pulls it in
transitively via ``sklearn.metrics`` (chain_replay_ml/training/evaluator.py).

The runtime hook ``pyi_hooks/rth_scipy_distn_fix.py`` source-patches this
module before it executes, which requires the real .py source to actually
be bundled alongside the .pyc (PyInstaller normally only ships bytecode).
Setting ``module_collection_mode = "pyz+py"`` here forces that.
"""

module_collection_mode = "pyz+py"
