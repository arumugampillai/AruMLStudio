# Feature Identity

| Concept | Rule |
|---------|------|
| UUID | `^FEAT_[0-9A-F]{32}$` + parsed UUIDv7 semantics |
| Canonical name | `^[a-z][a-z0-9_]*$`, unique, immutable |
| Versions | `definition_version` vs `implementation_version` |
| Definition hash | name \| warmup \| gap \| memory \| sorted primitives (UTF-8 SHA-256) |
| Transform ref | `NULL` or `TR_[0-9A-F]{32}` |
| Lifecycle | EXPERIMENTAL → CANDIDATE → VALIDATED → DEPRECATED (stored only) |
