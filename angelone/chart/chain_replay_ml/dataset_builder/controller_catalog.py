"""Controller Registry — internal architecture layer for controller metadata.

Describes CONTROLLERS (logical market models), not features.

Architecture metadata and query source of truth; runtime authority remains
the legacy controller maps (CONTROLLER_REGISTRY / CONTROLLER_FEATURES) during
this transition. Bootstrapped from those tables so behaviour is unchanged.

Does NOT change calculations, Feature Registry schema, Transformation
Pipeline, datasets, models, or UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Iterable, Mapping

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Governance state — registration vs production visibility (filtering optional).
CONTROLLER_STATE_ACTIVE = "Active"
CONTROLLER_STATE_EXPERIMENTAL = "Experimental"
CONTROLLER_STATE_DEPRECATED = "Deprecated"
CONTROLLER_STATE_INTERNAL = "Internal"
CONTROLLER_STATES: frozenset[str] = frozenset(
    {
        CONTROLLER_STATE_ACTIVE,
        CONTROLLER_STATE_EXPERIMENTAL,
        CONTROLLER_STATE_DEPRECATED,
        CONTROLLER_STATE_INTERNAL,
    }
)


@dataclass(frozen=True)
class WarmupPolicy:
    """Controller warmup declaration."""

    warmup_type: str  # Sample | Calendar | Session | Immediate
    warmup_value: str  # e.g. "9", "5m", "Session", "0"


@dataclass(frozen=True)
class ControllerDefinition:
    """Full metadata for one controller (architecture layer).

    Controllers emit the smallest complete set of canonical market-state
    values from which controller-specific derived features can be
    reconstructed. Packaging (ratios, lags, interactions, …) belongs to
    the Transformation Pipeline — not here as an emission contract.
    """

    controller_id: str
    name: str
    description: str
    owner: str
    version: str
    inputs: tuple[str, ...]
    emitted_features: tuple[str, ...]
    dependencies: tuple[str, ...]
    warmup_policy: WarmupPolicy
    lifecycle: str  # Reset | Continue | Immediate | …  (gap / stream policy)
    controller_type: str  # Rolling | Composite | Session | Calendar | …
    readiness_state: str  # ControllerReady | AlwaysValid | Immediate | …
    phase: int = 0
    logical_family: str = ""
    controller_state: str = CONTROLLER_STATE_ACTIVE  # Active | Experimental | …
    sample_fields: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)


# Architectural emission invariant (documentation / validation).
CONTROLLER_EMISSION_RULE = (
    "A controller emits the smallest complete set of canonical market-state "
    "values from which all controller-specific derived features can be "
    "reconstructed. Controllers must not emit lag, return, difference, "
    "ratio packaging, interaction, rolling engineering, z-score packaging, "
    "or experiment-only features."
)

# Packaging suffixes that Controllers should not emit as canonical outputs.
# EDGE leftovers may still appear in emitted_features until a later migration;
# validate() reports them as warnings, not hard failures.
_PACKAGING_SUFFIX_MARKERS: tuple[str, ...] = (
    "_to_ltp_ratio",
    "_to_spot_ratio",
    "_x_",
    "_lag_",
    "_return_",
    "_change_",
    "_zscore_",
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ControllerRegistry:
    """In-process registry of controller definitions.

    Thread-safe. Registration is idempotent when definitions are equal.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_id: dict[str, ControllerDefinition] = {}
        self._feature_owner: dict[str, str] = {}
        self._bootstrapped = False

    # -- registration -------------------------------------------------------

    def register(
        self,
        definition: ControllerDefinition,
        *,
        replace: bool = False,
    ) -> None:
        """Register a controller definition.

        Raises ``ValueError`` on id conflict (unless ``replace``) or when an
        emitted feature is already owned by a different controller.
        """
        with self._lock:
            existing = self._by_id.get(definition.controller_id)
            if existing is not None and not replace:
                if existing == definition:
                    return
                raise ValueError(
                    f"controller already registered: {definition.controller_id}"
                )
            self._index_definition(definition, replace=replace)

    def register_many(
        self,
        definitions: Iterable[ControllerDefinition],
        *,
        replace: bool = False,
    ) -> None:
        for definition in definitions:
            self.register(definition, replace=replace)

    def _index_definition(
        self,
        definition: ControllerDefinition,
        *,
        replace: bool,
    ) -> None:
        cid = definition.controller_id
        if replace and cid in self._by_id:
            old = self._by_id[cid]
            for feat in old.emitted_features:
                if self._feature_owner.get(feat) == cid:
                    del self._feature_owner[feat]

        for feat in definition.emitted_features:
            owner = self._feature_owner.get(feat)
            if owner is not None and owner != cid:
                raise ValueError(
                    f"feature {feat!r} already owned by {owner!r}; "
                    f"cannot assign to {cid!r}"
                )

        self._by_id[cid] = definition
        for feat in definition.emitted_features:
            self._feature_owner[feat] = cid

    # -- queries ------------------------------------------------------------

    def get(self, controller_id: str) -> ControllerDefinition | None:
        with self._lock:
            return self._by_id.get(controller_id)

    def require(self, controller_id: str) -> ControllerDefinition:
        definition = self.get(controller_id)
        if definition is None:
            raise KeyError(f"unknown controller: {controller_id}")
        return definition

    def all(self) -> tuple[ControllerDefinition, ...]:
        with self._lock:
            return tuple(self._by_id[cid] for cid in sorted(self._by_id))

    def controller_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._by_id)

    def owner_of_feature(self, feature: str) -> str | None:
        """Return controller_id that emits ``feature``, or None."""
        with self._lock:
            return self._feature_owner.get(str(feature or "").strip())

    def features_of(self, controller_id: str) -> tuple[str, ...]:
        definition = self.get(controller_id)
        if definition is None:
            return ()
        return definition.emitted_features

    def controllers_in_family(self, logical_family: str) -> tuple[ControllerDefinition, ...]:
        family = str(logical_family or "").strip()
        with self._lock:
            return tuple(
                d for d in self._by_id.values() if d.logical_family == family
            )

    def controllers_by_state(
        self,
        *states: str,
    ) -> tuple[ControllerDefinition, ...]:
        """Filter registered controllers by ``controller_state``.

        Does not hide anything from emission today — callers choose when to
        apply production vs experimental filters.
        """
        wanted = {str(s) for s in states} if states else {CONTROLLER_STATE_ACTIVE}
        with self._lock:
            return tuple(
                d for d in sorted(self._by_id.values(), key=lambda x: x.controller_id)
                if d.controller_state in wanted
            )

    def is_bootstrapped(self) -> bool:
        with self._lock:
            return self._bootstrapped

    def mark_bootstrapped(self) -> None:
        with self._lock:
            self._bootstrapped = True

    # -- validation ---------------------------------------------------------

    def validate(self) -> list[str]:
        """Return human-readable issues (empty ⇒ healthy).

        Hard problems (missing deps, cycles, duplicate feature owners) are
        always reported. Packaging markers in ``emitted_features`` are
        reported as warnings (EDGE leftovers allowed until migrated).
        """
        issues: list[str] = []
        with self._lock:
            ids = set(self._by_id)
            # dependency existence
            for cid, definition in self._by_id.items():
                if definition.controller_state not in CONTROLLER_STATES:
                    issues.append(
                        f"{cid}: unknown controller_state {definition.controller_state!r}"
                    )
                for dep in definition.dependencies:
                    if dep not in ids:
                        issues.append(f"{cid}: missing dependency {dep!r}")

            # reverse-index consistency
            for feat, owner in self._feature_owner.items():
                definition = self._by_id.get(owner)
                if definition is None:
                    issues.append(f"feature {feat!r}: owner {owner!r} not registered")
                elif feat not in definition.emitted_features:
                    issues.append(
                        f"feature {feat!r}: indexed to {owner!r} but not in emitted_features"
                    )

            # packaging warnings (non-fatal architectural drift)
            for cid, definition in self._by_id.items():
                for feat in definition.emitted_features:
                    if any(m in feat for m in _PACKAGING_SUFFIX_MARKERS):
                        issues.append(
                            f"warning: {cid} emits packaging-like feature {feat!r} "
                            f"({CONTROLLER_EMISSION_RULE[:60]}…)"
                        )

            # cycles
            cycles = self._detect_cycles_unlocked()
            for cycle in cycles:
                issues.append(f"dependency cycle: {' -> '.join(cycle)}")

        return issues

    def _detect_cycles_unlocked(self) -> list[list[str]]:
        graph = {cid: list(d.dependencies) for cid, d in self._by_id.items()}
        cycles: list[list[str]] = []
        visited: set[str] = set()
        stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> None:
            if node in stack:
                if node in path:
                    i = path.index(node)
                    cycles.append(path[i:] + [node])
                return
            if node in visited:
                return
            visited.add(node)
            stack.add(node)
            path.append(node)
            for dep in graph.get(node, []):
                if dep in graph:
                    dfs(dep)
            path.pop()
            stack.remove(node)

        for cid in graph:
            dfs(cid)
        return cycles


# Module singleton — populated by bootstrap / controller init.
_DEFAULT_REGISTRY = ControllerRegistry()


def get_controller_registry() -> ControllerRegistry:
    """Return the process-wide Controller Registry."""
    return _DEFAULT_REGISTRY


def reset_controller_registry_for_tests() -> None:
    """Clear the default registry (tests only)."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = ControllerRegistry()
