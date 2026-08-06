"""Bootstrap Controller Registry from existing controller_registry tables.

Preserves all IDs, warmup, dependencies, and emitted-feature mappings.
Adds display metadata (name, description, lifecycle, family) without
changing runtime emission behaviour.
"""

from __future__ import annotations

from .controller_catalog import (
    CONTROLLER_STATE_ACTIVE,
    ControllerDefinition,
    ControllerRegistry,
    WarmupPolicy,
    get_controller_registry,
)
from .controller_registry import (
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
    FEATURE_REGISTRY_VERSION,
    ControllerSpec,
)

# Logical families — documentation / grouping only (not new controller IDs).
LOGICAL_FAMILY_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("token.ltp.", "LtpController"),
    ("token.book", "MarketMicrostructureController"),
    ("token.chain", "ChainController"),
    ("composite.weighted_ltp_ema", "WeightedEMAController"),
    ("token.rv.", "RVController"),
    ("token.iv_window.", "IVController"),
    ("token.iv_history.", "IVController"),
    ("token.iv.", "IVController"),
    ("token.roll", "IVController"),
    ("token.dgt", "IVController"),
    ("spot.hl.", "SpotHLController"),
    ("spot.rv.", "RVController"),
    ("spot.momentum", "SpotController"),
    ("composite.iv_rv_spread", "RVController"),
    ("composite.weighted_spot_ema", "WeightedEMAController"),
    ("composite.weighted_spot_hl", "WeightedEMAController"),
    ("composite.iv_x_spot_ema", "WeightedEMAController"),
    ("spot.", "SpotController"),
)

_LIFECYCLE_BY_FAMILY: dict[str, str] = {
    "LtpController": "Reset",
    "MarketMicrostructureController": "Reset",
    "ChainController": "Reset",
    "SpotController": "Continue",
    "SpotHLController": "Continue",
    "IVController": "Continue",
    "RVController": "Continue",
    "WeightedEMAController": "Continue",
}

_TYPE_BY_WARMUP: dict[str, str] = {
    "Sample": "Rolling",
    "Calendar": "Calendar",
    "Session": "Session",
    "Immediate": "Immediate",
}

_READINESS_BY_WARMUP: dict[str, str] = {
    "Sample": "ControllerReady",
    "Calendar": "ControllerReady",
    "Session": "ControllerReady",
    "Immediate": "Immediate",
}


def logical_family_for(controller_id: str) -> str:
    for prefix, family in LOGICAL_FAMILY_BY_PREFIX:
        if controller_id.startswith(prefix) or controller_id == prefix:
            return family
    return "Controller"


def _display_name(controller_id: str) -> str:
    return controller_id.replace(".", " ").replace("_", " ").title()


def _description(spec: ControllerSpec, features: tuple[str, ...]) -> str:
    feat_preview = ", ".join(features[:4]) if features else "(no registry emissions)"
    if len(features) > 4:
        feat_preview += ", …"
    return (
        f"Phase-{spec.phase} controller {spec.controller_id}; "
        f"warmup {spec.warmup_type}/{spec.warmup_value}; "
        f"emits [{feat_preview}]"
    )


def definition_from_spec(spec: ControllerSpec) -> ControllerDefinition:
    features = tuple(CONTROLLER_FEATURES.get(spec.controller_id, []) or ())
    family = logical_family_for(spec.controller_id)
    is_composite = spec.controller_id.startswith("composite.")
    controller_type = (
        "Composite"
        if is_composite
        else _TYPE_BY_WARMUP.get(spec.warmup_type, "Rolling")
    )
    return ControllerDefinition(
        controller_id=spec.controller_id,
        name=_display_name(spec.controller_id),
        description=_description(spec, features),
        owner="dataset_builder",
        version=str(FEATURE_REGISTRY_VERSION),
        inputs=tuple(spec.sample_fields),
        emitted_features=features,
        dependencies=tuple(spec.source_controllers),
        warmup_policy=WarmupPolicy(spec.warmup_type, spec.warmup_value),
        lifecycle=_LIFECYCLE_BY_FAMILY.get(family, "Continue"),
        controller_type=controller_type,
        readiness_state=_READINESS_BY_WARMUP.get(spec.warmup_type, "ControllerReady"),
        phase=int(spec.phase),
        logical_family=family,
        controller_state=CONTROLLER_STATE_ACTIVE,
        sample_fields=tuple(spec.sample_fields),
        metadata={
            "source": "controller_registry.CONTROLLER_REGISTRY",
            "feature_registry_version": str(FEATURE_REGISTRY_VERSION),
        },
    )


def all_bootstrapped_definitions() -> tuple[ControllerDefinition, ...]:
    return tuple(
        definition_from_spec(CONTROLLER_REGISTRY[cid])
        for cid in sorted(CONTROLLER_REGISTRY)
    )


def bootstrap_controller_registry(
    registry: ControllerRegistry | None = None,
    *,
    replace: bool = True,
) -> ControllerRegistry:
    """Load every legacy ControllerSpec into the Controller Registry.

    Idempotent. Safe to call from module import / controller init.
    """
    reg = registry if registry is not None else get_controller_registry()
    if reg.is_bootstrapped() and not replace:
        return reg
    reg.register_many(all_bootstrapped_definitions(), replace=replace)
    reg.mark_bootstrapped()
    return reg


def ensure_controller_registry() -> ControllerRegistry:
    """Ensure the default registry is bootstrapped; return it."""
    reg = get_controller_registry()
    if not reg.is_bootstrapped():
        bootstrap_controller_registry(reg, replace=True)
    return reg


def owner_of_feature(feature: str) -> str | None:
    """Feature Registry helper: which controller owns this Computed Base feature."""
    return ensure_controller_registry().owner_of_feature(feature)


def validate_controller_registry() -> list[str]:
    """Run registry validation (hard issues + packaging warnings)."""
    return ensure_controller_registry().validate()
