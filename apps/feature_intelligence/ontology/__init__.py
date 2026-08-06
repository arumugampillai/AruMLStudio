"""Feature Ontology package — Sprint 6 (classification metadata only; no AI)."""

from __future__ import annotations

from feature_intelligence.ontology.catalog import (
    EXPECTED_ONTOLOGY_SEED_HASH,
    EXPECTED_VOCAB_SEED_HASH,
    ONTOLOGY_VERSION,
    VOCAB_PACK_VERSION,
    SEED_ONTOLOGY_ROWS,
    SEED_VOCABULARIES,
    compute_ontology_seed_hash,
    compute_vocab_seed_hash,
)
from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.import_export import export_ontology, import_ontology
from feature_intelligence.ontology.models import (
    OBJECT_TYPE_TABLE,
    CoverageReport,
    OntologyRecord,
    VocabularyRecord,
    normalize_id_list,
)
from feature_intelligence.ontology.service import (
    OntologyNotFoundError,
    OntologyService,
)
from feature_intelligence.ontology.store import OntologyStore

__all__ = [
    "ONTOLOGY_VERSION",
    "VOCAB_PACK_VERSION",
    "EXPECTED_VOCAB_SEED_HASH",
    "EXPECTED_ONTOLOGY_SEED_HASH",
    "SEED_VOCABULARIES",
    "SEED_ONTOLOGY_ROWS",
    "OBJECT_TYPE_TABLE",
    "VocabularyRecord",
    "OntologyRecord",
    "CoverageReport",
    "OntologyStore",
    "OntologyService",
    "OntologyNotFoundError",
    "derive_ontology_uuid",
    "normalize_id_list",
    "compute_vocab_seed_hash",
    "compute_ontology_seed_hash",
    "export_ontology",
    "import_ontology",
]
