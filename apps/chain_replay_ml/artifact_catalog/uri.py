"""Universal Artifact URI helpers (`aruneo://…`)."""

from __future__ import annotations

import re
from urllib.parse import quote, unquote

SCHEME = "aruneo"

_FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SEG_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class ArtifactUriError(ValueError):
    pass


def mint_uri(family: str, *parts: str) -> str:
    """Build ``aruneo://{family}/{part}/…`` with path-safe segments."""
    fam = str(family or "").strip().lower()
    if not _FAMILY_RE.match(fam):
        raise ArtifactUriError(f"invalid artifact family: {family!r}")
    segs: list[str] = []
    for p in parts:
        raw = str(p or "").strip()
        if not raw:
            raise ArtifactUriError("empty URI segment")
        # Allow richer ids but encode unsafe chars.
        if _SEG_RE.match(raw):
            segs.append(raw)
        else:
            segs.append(quote(raw, safe="._-"))
    if not segs:
        raise ArtifactUriError("URI requires at least one identity segment")
    return f"{SCHEME}://{fam}/{'/'.join(segs)}"


def parse_uri(uri: str) -> tuple[str, list[str]]:
    """Return (family, segments)."""
    text = str(uri or "").strip()
    prefix = f"{SCHEME}://"
    if not text.startswith(prefix):
        raise ArtifactUriError(f"URI must start with {prefix}: {uri!r}")
    rest = text[len(prefix) :]
    bits = [unquote(b) for b in rest.split("/") if b]
    if len(bits) < 2:
        raise ArtifactUriError(f"URI needs family + identity: {uri!r}")
    family, *segs = bits
    if not _FAMILY_RE.match(family):
        raise ArtifactUriError(f"invalid family in URI: {uri!r}")
    return family, segs


def is_artifact_uri(uri: str) -> bool:
    try:
        parse_uri(uri)
        return True
    except ArtifactUriError:
        return False


def master_day_uri(trading_day: str) -> str:
    return mint_uri("master", "day", trading_day)


def training_uri(artifact_id: str, *extra: str) -> str:
    return mint_uri("training", artifact_id, *extra) if extra else mint_uri("training", artifact_id)


def model_uri(model_name: str) -> str:
    return mint_uri("model", model_name)


def feature_studio_uri(studio: str, model_name: str) -> str:
    return mint_uri("feature_studio", studio, model_name)


def diagnostics_uri(model_name: str) -> str:
    return mint_uri("diagnostics", model_name)


def experiment_uri(experiment_id: str) -> str:
    return mint_uri("experiment", experiment_id)


def eval_uri(experiment_id: str, result_id: str = "result") -> str:
    return mint_uri("eval", experiment_id, result_id)


def prediction_uri(name: str) -> str:
    return mint_uri("prediction", name)
