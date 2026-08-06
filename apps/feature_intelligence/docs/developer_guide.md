# Developer Guide

## Prerequisites

- Python 3.12+ (project default)
- `angelone/chart` on `PYTHONPATH` (PyCharm / launch scripts typically set this)
- Optional: `PyYAML` for full YAML parsing (stdlib lite loader is used as fallback)

## Initialize a local DB

```bash
cd angelone/chart
python -m feature_intelligence init-db --db %TEMP%\fic_dev.db
python -m feature_intelligence status --db %TEMP%\fic_dev.db
```

## Add a migration (later sprints)

1. Create `migrations/versions/0002_your_change.py`
2. Export `version`, `description`, `upgrade(conn)`, `downgrade(conn)`
3. Run `python -m feature_intelligence migrate`
4. Add a unit/integration test under `tests/`

Do **not** hand-edit `feature_intelligence.db` schema outside migrations.

## Run tests

```bash
cd angelone/chart
python -m unittest discover -s feature_intelligence/tests -v
```

## Logging

```python
from feature_intelligence.core import load_config, setup_logging, get_logger

cfg = load_config()
setup_logging(cfg.logging)
log = get_logger("my_module")
log.info("hello")
```

## Starting Sprint 1

Sprint 1 owns Primitive Intelligence (Step 0) under `registry/` (and related docs).  
Use this package's DB/migrations/config/logging — do not invent a parallel infra stack.
