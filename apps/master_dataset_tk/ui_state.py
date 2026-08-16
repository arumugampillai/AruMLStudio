"""Global UI State Persistence service for the ML Research Studio Tkinter app.

One reusable service so the whole application remembers the user's last
selections — dropdowns, checkboxes, radios, numeric/text inputs, notebook
tabs, tree selections, window geometry, paned-window sash positions — across
relaunches. There are no Save/Apply buttons anywhere: every ``bind_*``
helper below wires transparent, debounced auto-save on change, and panels
restore state once, right after their widgets are built.

Storage
-------
One JSON file, ``UIStateManager.default_path()``:

- Windows: ``%APPDATA%/AruNeo/ui_state_tk.json``
- Elsewhere: ``~/.aruneo/ui_state_tk.json``

This sits next to ``ml_research_studio.json`` (see :mod:`project_config`),
i.e. the same stable, project-independent config directory the app already
uses for the remembered chart/master-data folders — so UI state survives
switching project folders and reinstalling into a new working directory.

The document is a *flat* dict of dotted, namespaced string keys, e.g.::

    {"create_model.dataset": "NIFTY_2026", "model_lab.tb_enable": true, ...}

Namespacing (``"<screen>.<control>"``) is a convention enforced by callers,
not by the storage format — this keeps ``get``/``set`` trivial and avoids
any nested-dict traversal / merge subtleties, while still guaranteeing two
unrelated screens can never collide or clobber each other's values as long
as each screen uses its own prefix.

Usage
-----
    from .ui_state import get_ui_state_manager

    state = get_ui_state_manager()
    state.bind_combobox(self._dataset_combo, "create_model.dataset", var=self._dataset_var)
    state.bind_checkbutton(chk, "create_model.use_gpu", var=self._gpu_var)
    state.bind_notebook(self._notebook, "create_model.tab")
    state.restore_window(self, "app.main_window")

For Dataset/Model dropdowns that must fall back to the *newest* item when
the saved selection no longer exists, keep using
``selection_lists.refresh_combobox`` / ``pick_preserved_or_default`` for the
list-population step (pass ``current=state.get(key)`` or use
``state.preferred_selection(key, items)``), then call ``bind_combobox``
with ``restore=False`` for ongoing autosave only — see call sites across
``master_dataset_tk`` for the exact pattern.

Safety
------
Every public method swallows and ignores errors from missing widgets,
destroyed Tk windows, corrupt JSON, and missing keys — this service must
never be the reason the UI crashes or fails to start.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Callable, Sequence

DEFAULT_STORAGE_FILE = "ui_state_tk.json"
DEFAULT_DEBOUNCE_MS = 400
_SENTINEL = object()


def default_settings_path() -> str:
    """Stable, per-user, project-independent path for the UI state file.

    Override with ``ARUMLSTUDIO_UI_STATE_PATH`` or ``ARUNEO_UI_STATE_PATH``
    (primarily for tests / isolated environments — never set in normal app usage).
    """
    override = str(
        os.environ.get("ARUMLSTUDIO_UI_STATE_PATH")
        or os.environ.get("ARUNEO_UI_STATE_PATH")
        or ""
    ).strip()
    if override:
        return override
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "AruMLStudio") if os.environ.get("APPDATA") else os.path.join(base, ".arumlstudio")
    os.makedirs(folder, exist_ok=True)
    target_path = os.path.join(folder, DEFAULT_STORAGE_FILE)
    if not os.path.isfile(target_path):
        legacy_folder = os.path.join(base, "AruNeo") if os.environ.get("APPDATA") else os.path.join(base, ".aruneo")
        legacy_path = os.path.join(legacy_folder, DEFAULT_STORAGE_FILE)
        if os.path.isfile(legacy_path):
            try:
                import shutil
                shutil.copy2(legacy_path, target_path)
            except Exception:
                pass
    return target_path


class UIStateManager:
    """Loads/saves one flat JSON document and wires Tk widgets to it.

    Not thread-safe for concurrent *writers* from multiple threads calling
    ``set`` at the exact same instant is fine (protected by a lock), but all
    ``bind_*`` helpers are meant to be used from the Tk main thread, as with
    any other Tkinter widget interaction.
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        autoload: bool = True,
    ) -> None:
        self.path = path or default_settings_path()
        self._debounce_ms = max(0, int(debounce_ms))
        self._data: dict[str, Any] = {}
        self._dirty = False
        self._lock = threading.RLock()
        self._scheduler: Any = None
        self._after_id: Any = None
        self._save_count = 0
        if autoload:
            self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> dict[str, Any]:
        """(Re)load the on-disk document into memory. Never raises."""
        with self._lock:
            self._data = self._read_disk()
            self._dirty = False
            return dict(self._data)

    def _read_disk(self) -> dict[str, Any]:
        try:
            if not os.path.isfile(self.path):
                return {}
            with open(self.path, encoding="utf-8") as fh:
                doc = json.load(fh)
            return doc if isinstance(doc, dict) else {}
        except (OSError, ValueError, TypeError):
            # Missing file, permission error, or corrupt/non-JSON contents —
            # fall back to an empty document rather than crashing the app.
            return {}
        except Exception:
            return {}

    def save(self) -> None:
        """Force an immediate flush to disk, bypassing any debounce timer."""
        self._cancel_pending()
        self._flush()

    def flush(self) -> None:
        """Alias for :meth:`save` — convenient on app exit."""
        self.save()

    def _flush(self) -> None:
        with self._lock:
            self._after_id = None
            if not self._dirty:
                return
            try:
                on_disk = self._read_disk()
                on_disk.update(self._data)
                folder = os.path.dirname(self.path)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                tmp_path = f"{self.path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(on_disk, fh, indent=2, sort_keys=True)
                os.replace(tmp_path, self.path)
                self._data = on_disk
                self._dirty = False
                self._save_count += 1
            except Exception:
                # Never let a settings-file write crash the UI thread.
                pass

    @property
    def save_count(self) -> int:
        """Number of times this manager has actually written to disk (tests)."""
        return self._save_count

    # ------------------------------------------------------------------
    # get / set
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
        *,
        widget: Any = None,
        debounce: bool = True,
        debounce_ms: int | None = None,
    ) -> None:
        """Update ``key`` in memory and schedule a debounced disk write.

        ``widget`` — any live Tk widget (has ``.after`` / ``.after_cancel``)
        used to schedule the save on the Tk event loop without blocking the
        UI. If omitted, the manager falls back to the widget registered via
        :meth:`attach_root`, or an immediate synchronous write when no Tk
        context is available at all (e.g. headless callers/tests).
        """
        with self._lock:
            if key in self._data and self._data[key] == value:
                return
            self._data[key] = value
            self._dirty = True
        self._schedule_save(widget=widget, debounce=debounce, debounce_ms=debounce_ms)

    def attach_root(self, root: Any) -> None:
        """Register the app's root Tk widget as the default save scheduler."""
        self._scheduler = root

    def _schedule_save(
        self,
        *,
        widget: Any = None,
        debounce: bool = True,
        debounce_ms: int | None = None,
    ) -> None:
        scheduler = widget if widget is not None else self._scheduler
        if scheduler is None or not debounce:
            self._flush()
            return
        delay = self._debounce_ms if debounce_ms is None else max(0, int(debounce_ms))
        self._cancel_pending(scheduler)
        try:
            self._after_id = scheduler.after(delay, self._flush)
            self._scheduler = scheduler
        except Exception:
            self._flush()

    def _cancel_pending(self, scheduler: Any = None) -> None:
        scheduler = scheduler if scheduler is not None else self._scheduler
        after_id = self._after_id
        self._after_id = None
        if after_id is not None and scheduler is not None:
            try:
                scheduler.after_cancel(after_id)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Dataset / Model preserve-or-newest helper
    # ------------------------------------------------------------------
    def preferred_selection(self, key: str, items: Sequence[str], *, default: str | None = None) -> str:
        """Saved value if still present in ``items`` (newest-first); else newest.

        Thin convenience wrapper around ``selection_lists.pick_preserved_or_default``
        so panels don't need to import both modules for the common
        "Dataset/Model combobox" pattern.
        """
        from .selection_lists import pick_preserved_or_default

        current = self.get(key, default)
        return pick_preserved_or_default(items, current)

    # ------------------------------------------------------------------
    # Per-project namespacing
    # ------------------------------------------------------------------
    @staticmethod
    def scope_key(project_path: str, name: str) -> str:
        """Build a per-project namespaced key, e.g. for chart-dir-scoped prefs.

        The global store is one flat document shared by every project the
        user has opened, so panels whose settings are meaningful only for a
        specific chart/data folder (previously their own ``*_prefs*.json``
        next to that folder) must not let two different projects clobber
        each other's saved values under the same short key. This hashes the
        normalized project path to a short stable tag and folds it into the
        dotted key, keeping keys short while remaining unique per project.
        """
        norm = os.path.normcase(os.path.normpath(str(project_path or "")))
        tag = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:10]
        return f"{name}@{tag}"

    # ------------------------------------------------------------------
    # Widget bind helpers
    # ------------------------------------------------------------------
    def bind_combobox(
        self,
        combo: Any,
        key: str,
        *,
        var: Any = None,
        default: str = "",
        restore: bool = True,
        widget: Any = None,
    ) -> None:
        """Auto-save a ``ttk.Combobox`` selection; optionally restore it now.

        Pass ``restore=False`` when the caller already restored the value
        itself (e.g. via ``refresh_combobox(..., current=state.get(key))``)
        and only wants ongoing autosave wired up.
        """
        if restore:
            saved = self.get(key, default)
            self._safe_set_text(var, combo, saved)

        def _on_change(*_args: Any) -> None:
            value = self._safe_get_text(var, combo)
            if value is None:
                return
            self.set(key, value, widget=widget or combo)

        self._bind_change(var, combo, "<<ComboboxSelected>>", _on_change)

    def bind_checkbutton(
        self,
        widget: Any,
        key: str,
        *,
        var: Any,
        default: bool = False,
        restore: bool = True,
    ) -> None:
        """Auto-save a ``ttk.Checkbutton``'s boolean variable."""
        if restore:
            saved = self.get(key, default)
            try:
                var.set(bool(saved))
            except Exception:
                pass

        def _on_change(*_args: Any) -> None:
            try:
                value = bool(var.get())
            except Exception:
                return
            self.set(key, value, widget=widget)

        self._trace_var(var, _on_change)

    def bind_radiobutton(
        self,
        var: Any,
        key: str,
        *,
        default: Any = None,
        restore: bool = True,
        widget: Any = None,
    ) -> None:
        """Auto-save a Radiobutton *group*'s shared variable (Int/StringVar).

        Radiobuttons in a group share one Tk variable, so bind the group's
        variable once (not each individual ``ttk.Radiobutton``).
        """
        if restore:
            saved = self.get(key, _SENTINEL)
            if saved is not _SENTINEL:
                try:
                    var.set(saved)
                except Exception:
                    pass
            elif default is not None:
                try:
                    var.set(default)
                except Exception:
                    pass

        def _on_change(*_args: Any) -> None:
            try:
                value = var.get()
            except Exception:
                return
            self.set(key, value, widget=widget)

        self._trace_var(var, _on_change)

    def bind_entry(
        self,
        widget: Any,
        key: str,
        *,
        var: Any = None,
        default: str = "",
        restore: bool = True,
        debounce_ms: int | None = None,
    ) -> None:
        """Auto-save an ``Entry``/``Spinbox`` value (debounced while typing)."""
        if restore:
            saved = self.get(key, default)
            self._safe_set_text(var, widget, saved, is_entry=True)

        def _on_change(*_args: Any) -> None:
            value = self._safe_get_text(var, widget)
            if value is None:
                return
            self.set(key, value, widget=widget, debounce_ms=debounce_ms)

        if var is not None and hasattr(var, "trace_add"):
            self._trace_var(var, _on_change)
        else:
            try:
                widget.bind("<KeyRelease>", _on_change, add="+")
            except Exception:
                pass

    # Spinboxes are entry-like widgets from the persistence layer's POV.
    def bind_spinbox(
        self,
        widget: Any,
        key: str,
        *,
        var: Any = None,
        default: str = "",
        restore: bool = True,
        debounce_ms: int | None = None,
    ) -> None:
        self.bind_entry(widget, key, var=var, default=default, restore=restore, debounce_ms=debounce_ms)

    def bind_notebook(
        self,
        notebook: Any,
        key: str,
        *,
        default_index: int = 0,
        restore: bool = True,
    ) -> None:
        """Auto-save the selected tab *index* of a ``ttk.Notebook``."""
        if restore:
            idx = self.get(key, default_index)
            try:
                tabs = notebook.tabs()
                if isinstance(idx, int) and 0 <= idx < len(tabs):
                    notebook.select(idx)
            except Exception:
                pass

        def _on_change(_evt: Any = None) -> None:
            try:
                idx = notebook.index(notebook.select())
            except Exception:
                return
            self.set(key, idx, widget=notebook)

        try:
            notebook.bind("<<NotebookTabChanged>>", _on_change, add="+")
        except Exception:
            pass

    def bind_tree_selection(
        self,
        tree: Any,
        key: str,
        *,
        default: str | None = None,
        restore: bool = True,
    ) -> None:
        """Auto-save/restore a Treeview's selected row ``iid`` (best-effort).

        Restoration only re-selects the row if that ``iid`` is still present
        (rows are frequently rebuilt from live data) — a missing row is
        silently ignored rather than raising.
        """
        if restore:
            saved = self.get(key, default)
            if saved:
                try:
                    if tree.exists(saved):
                        tree.selection_set(saved)
                        tree.see(saved)
                except Exception:
                    pass

        def _on_change(_evt: Any = None) -> None:
            try:
                sel = tree.selection()
            except Exception:
                return
            value = sel[0] if sel else None
            self.set(key, value, widget=tree)

        try:
            tree.bind("<<TreeviewSelect>>", _on_change, add="+")
        except Exception:
            pass

    def bind_panedwindow(self, paned: Any, key: str, *, restore: bool = True) -> None:
        """Auto-save/restore ``ttk.Panedwindow`` sash positions."""

        def _apply_saved() -> None:
            saved = self.get(key)
            if not isinstance(saved, list):
                return
            try:
                pane_count = len(paned.panes())
            except Exception:
                return
            for i, pos in enumerate(saved):
                if i >= max(0, pane_count - 1):
                    break
                try:
                    paned.sashpos(i, int(pos))
                except Exception:
                    pass

        if restore:
            try:
                paned.after(50, _apply_saved)
            except Exception:
                _apply_saved()

        def _on_release(_evt: Any = None) -> None:
            try:
                pane_count = len(paned.panes())
                positions = [paned.sashpos(i) for i in range(max(0, pane_count - 1))]
            except Exception:
                return
            self.set(key, positions, widget=paned, debounce_ms=600)

        try:
            paned.bind("<ButtonRelease-1>", _on_release, add="+")
        except Exception:
            pass

    def restore_window(
        self,
        window: Any,
        key: str,
        *,
        default_geometry: str | None = None,
        restore: bool = True,
    ) -> None:
        """Auto-save/restore a window's ``geometry()`` string."""
        if restore:
            saved = self.get(key)
            geom = saved.get("geometry") if isinstance(saved, dict) else None
            geom = geom or default_geometry
            if geom:
                try:
                    window.geometry(geom)
                except Exception:
                    pass

        def _on_configure(_evt: Any = None) -> None:
            try:
                geom = window.geometry()
            except Exception:
                return
            self.set(key, {"geometry": geom}, widget=window, debounce_ms=800)

        try:
            window.bind("<Configure>", _on_configure, add="+")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal widget helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _trace_var(var: Any, callback: Callable[..., None]) -> None:
        try:
            var.trace_add("write", callback)
        except Exception:
            pass

    @staticmethod
    def _bind_change(var: Any, widget: Any, sequence: str, callback: Callable[..., None]) -> None:
        if var is not None and hasattr(var, "trace_add"):
            UIStateManager._trace_var(var, callback)
            return
        try:
            widget.bind(sequence, callback, add="+")
        except Exception:
            pass

    @staticmethod
    def _safe_get_text(var: Any, widget: Any) -> Any:
        try:
            if var is not None:
                return var.get()
            return widget.get()
        except Exception:
            return None

    @staticmethod
    def _safe_set_text(var: Any, widget: Any, value: Any, *, is_entry: bool = False) -> None:
        try:
            if var is not None:
                var.set(value)
                return
            if is_entry:
                widget.delete(0, "end")
                widget.insert(0, value)
            else:
                widget.set(value)
        except Exception:
            pass


# ----------------------------------------------------------------------
# Process-wide singleton
# ----------------------------------------------------------------------
_singleton: UIStateManager | None = None
_singleton_lock = threading.Lock()


def get_ui_state_manager() -> UIStateManager:
    """Return the process-wide :class:`UIStateManager` singleton (lazy init)."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = UIStateManager()
        return _singleton


def set_ui_state_manager(manager: UIStateManager | None) -> None:
    """Override the process-wide singleton — for tests / isolated embedding."""
    global _singleton
    with _singleton_lock:
        _singleton = manager


__all__ = [
    "DEFAULT_STORAGE_FILE",
    "DEFAULT_DEBOUNCE_MS",
    "UIStateManager",
    "default_settings_path",
    "get_ui_state_manager",
    "set_ui_state_manager",
]
