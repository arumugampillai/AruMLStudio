"""CLI for FIC infrastructure through Semantic Query (Sprint 0–9)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature_intelligence",
        description="Feature Intelligence Core — CLI (Sprint 0–9)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Create DB and apply migrations")
    p_init.add_argument("--db", help="Override database path")

    p_mig = sub.add_parser("migrate", help="Apply or roll back migrations")
    p_mig.add_argument("--db", help="Override database path")
    p_mig.add_argument("--downgrade", action="store_true")
    p_mig.add_argument("--steps", type=int, default=1)

    p_status = sub.add_parser("status", help="Show migration status")
    p_status.add_argument("--db", help="Override database path")

    def _add_db(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", help="Override database path")

    p_prim = sub.add_parser("primitives", help="Primitive Catalog commands")
    prim_sub = p_prim.add_subparsers(dest="primitives_cmd", required=True)
    _add_db(prim_sub.add_parser("list"))
    p_get = prim_sub.add_parser("get")
    p_get.add_argument("--id", required=True, dest="primitive_id")
    _add_db(p_get)
    _add_db(prim_sub.add_parser("validate"))

    p_feat = sub.add_parser(
        "features", help="Feature Registry commands (read-only + i/o + sync)"
    )
    feat_sub = p_feat.add_subparsers(dest="features_cmd", required=True)
    _add_db(feat_sub.add_parser("list"))
    p_fget = feat_sub.add_parser("get")
    p_fget.add_argument("--id", dest="feature_uuid")
    p_fget.add_argument("--name", dest="canonical_name")
    _add_db(p_fget)
    _add_db(feat_sub.add_parser("validate"))
    p_exp = feat_sub.add_parser("export")
    p_exp.add_argument("--format", choices=("json", "yaml"), default="json")
    p_exp.add_argument("--out", required=True)
    _add_db(p_exp)
    p_imp = feat_sub.add_parser("import")
    p_imp.add_argument("--format", choices=("json", "yaml"), default="json")
    p_imp.add_argument("--in", dest="infile", required=True)
    _add_db(p_imp)
    p_fsync = feat_sub.add_parser(
        "sync",
        help="Sync legacy Feature Registry catalog into FIC feature_registry",
    )
    p_fsync.add_argument(
        "--data-dir",
        dest="data_dir",
        help="Chart data directory (contains feature_registry_store.json)",
    )
    p_fsync.add_argument(
        "--chart-dir",
        dest="chart_dir",
        help="Chart root; data dir defaults to <chart-dir>/data",
    )
    p_fsync.add_argument(
        "--mode",
        choices=("strict", "lenient"),
        default="lenient",
        help="Primitive mapping mode (default: lenient)",
    )
    p_fsync.add_argument(
        "--force",
        action="store_true",
        help="Overwrite definition when hash conflicts",
    )
    p_fsync.add_argument(
        "--research-sync",
        dest="research_sync",
        action="store_true",
        default=True,
        help="After import, sync research FRR shells (default: on)",
    )
    p_fsync.add_argument(
        "--no-research-sync",
        dest="research_sync",
        action="store_false",
        help="Skip research FRR sync after feature import",
    )
    _add_db(p_fsync)

    p_ops = sub.add_parser("operators", help="Operator Registry commands (read-only + i/o)")
    ops_sub = p_ops.add_subparsers(dest="operators_cmd", required=True)
    _add_db(ops_sub.add_parser("list"))
    p_oget = ops_sub.add_parser("get")
    p_oget.add_argument("--id", dest="operator_id")
    p_oget.add_argument("--name", dest="canonical_name")
    _add_db(p_oget)
    _add_db(ops_sub.add_parser("validate"))
    p_oexp = ops_sub.add_parser("export")
    p_oexp.add_argument("--format", choices=("json", "yaml"), default="json")
    p_oexp.add_argument("--out", required=True)
    _add_db(p_oexp)
    p_oimp = ops_sub.add_parser("import")
    p_oimp.add_argument("--format", choices=("json", "yaml"), default="json")
    p_oimp.add_argument("--in", dest="infile", required=True)
    _add_db(p_oimp)

    p_gram = sub.add_parser("grammar", help="Transformation Language (Sprint 4)")
    gram_sub = p_gram.add_subparsers(dest="grammar_cmd", required=True)
    gram_sub.add_parser("version")
    p_gv = gram_sub.add_parser("validate")
    p_gv.add_argument("--mode", choices=("syntax_only", "bound"), default="syntax_only")
    p_gv.add_argument("--file", dest="expr_file")
    p_gv.add_argument("--expr", dest="expr_text")
    _add_db(p_gv)
    p_gf = gram_sub.add_parser("format")
    p_gf.add_argument("--file", dest="expr_file")
    p_gf.add_argument("--expr", dest="expr_text")
    p_gexp = gram_sub.add_parser("export")
    p_gexp.add_argument("--format", choices=("json", "yaml", "text"), default="json")
    p_gexp.add_argument("--out", required=True)
    p_gexp.add_argument("--file", dest="expr_file")
    p_gexp.add_argument("--expr", dest="expr_text")
    p_gimp = gram_sub.add_parser("import")
    p_gimp.add_argument("--format", choices=("json", "yaml", "text"), default="json")
    p_gimp.add_argument("--in", dest="infile", required=True)
    p_gimp.add_argument("--out", required=True)

    def _add_expr(p: argparse.ArgumentParser) -> None:
        p.add_argument("--file", dest="expr_file")
        p.add_argument("--expr", dest="expr_text")
        _add_db(p)

    p_comp = sub.add_parser("compiler", help="Transformation Compiler (Sprint 5)")
    comp_sub = p_comp.add_subparsers(dest="compiler_cmd", required=True)
    p_cc = comp_sub.add_parser("compile")
    _add_expr(p_cc)
    p_cc.add_argument("--feature-uuid", dest="feature_uuid")
    p_cc.add_argument("--metrics", action="store_true")
    p_cc.add_argument("--no-cache-hit-event", action="store_true")
    p_cv = comp_sub.add_parser("validate")
    _add_expr(p_cv)
    p_cv.add_argument("--roundtrip", action="store_true")
    p_cm = comp_sub.add_parser("manifest")
    _add_expr(p_cm)
    p_ce = comp_sub.add_parser("export")
    p_ce.add_argument("--format", choices=("json", "yaml"), default="json")
    p_ce.add_argument("--out", required=True)
    _add_expr(p_ce)
    p_ci = comp_sub.add_parser("import")
    p_ci.add_argument("--format", choices=("json", "yaml"), default="json")
    p_ci.add_argument("--in", dest="infile", required=True)
    _add_db(p_ci)

    p_ont = sub.add_parser("ontology", help="Feature Ontology (Sprint 6)")
    ont_sub = p_ont.add_subparsers(dest="ontology_cmd", required=True)
    p_ol = ont_sub.add_parser("list")
    p_ol.add_argument(
        "--type",
        dest="object_type",
        choices=("PRIMITIVE", "OPERATOR", "TRANSFORMATION", "FEATURE"),
    )
    _add_db(p_ol)
    p_og = ont_sub.add_parser("get")
    p_og.add_argument("--type", dest="object_type", required=True,
                      choices=("PRIMITIVE", "OPERATOR", "TRANSFORMATION", "FEATURE"))
    p_og.add_argument("--id", dest="object_id", required=True)
    _add_db(p_og)
    p_ov = ont_sub.add_parser("validate")
    p_ov.add_argument("--mode", choices=("strict", "present"), default="strict")
    p_ov.add_argument("--strict-refs", action="store_true")
    _add_db(p_ov)
    _add_db(ont_sub.add_parser("coverage"))
    p_oe = ont_sub.add_parser("export")
    p_oe.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_oe.add_argument("--out", required=True)
    p_oe.add_argument(
        "--type",
        dest="object_type",
        choices=("PRIMITIVE", "OPERATOR", "TRANSFORMATION", "FEATURE"),
    )
    _add_db(p_oe)
    p_oi = ont_sub.add_parser("import")
    p_oi.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_oi.add_argument("--path", required=True)
    _add_db(p_oi)

    p_lin = sub.add_parser("lineage", help="Feature Lineage (Sprint 7)")
    lin_sub = p_lin.add_subparsers(dest="lineage_cmd", required=True)
    p_ll = lin_sub.add_parser("list")
    p_ll.add_argument("--relationship", dest="relationship_id")
    _add_db(p_ll)
    p_lg = lin_sub.add_parser("get")
    p_lg.add_argument("--id", dest="lineage_uuid")
    p_lg.add_argument("--parent", dest="parent_object")
    p_lg.add_argument("--child", dest="child_object")
    p_lg.add_argument("--relationship", dest="relationship_id")
    _add_db(p_lg)
    for nav in ("parents", "children", "ancestors", "descendants"):
        p_nav = lin_sub.add_parser(nav)
        p_nav.add_argument("--id", dest="object_id", required=True)
        _add_db(p_nav)
    p_lv = lin_sub.add_parser("validate")
    p_lv.add_argument("--mode", choices=("strict", "present"), default="strict")
    p_lv.add_argument("--strict-refs", action="store_true")
    _add_db(p_lv)
    p_ld = lin_sub.add_parser("derive")
    p_ld.add_argument("--transformation", dest="transformation_uuid")
    p_ld.add_argument("--feature", dest="feature_uuid")
    p_ld.add_argument("--no-closure", action="store_true")
    p_ld.add_argument("--strict-refs", action="store_true")
    _add_db(p_ld)
    _add_db(lin_sub.add_parser("stats"))
    p_le = lin_sub.add_parser("export")
    p_le.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_le.add_argument("--out", required=True)
    p_le.add_argument("--relationship", dest="relationship_id")
    _add_db(p_le)
    p_li = lin_sub.add_parser("import")
    p_li.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_li.add_argument("--path", required=True)
    _add_db(p_li)

    p_res = sub.add_parser("research", help="Feature Research Record (Sprint 8)")
    res_sub = p_res.add_subparsers(dest="research_cmd", required=True)
    p_rl = res_sub.add_parser("list")
    p_rl.add_argument(
        "--status",
        choices=("EMPTY", "ACTIVE", "ARCHIVED"),
    )
    _add_db(p_rl)
    p_rg = res_sub.add_parser("get")
    p_rg.add_argument("--id", dest="research_uuid")
    p_rg.add_argument("--feature", dest="feature_uuid")
    _add_db(p_rg)
    p_rv = res_sub.add_parser("validate")
    p_rv.add_argument("--mode", choices=("strict", "present"), default="strict")
    p_rv.add_argument("--strict-refs", action="store_true")
    p_rv.add_argument("--strict-coverage", action="store_true")
    _add_db(p_rv)
    p_rs = res_sub.add_parser("sync")
    p_rs.add_argument("--feature", dest="feature_uuid")
    _add_db(p_rs)
    _add_db(res_sub.add_parser("stats"))
    _add_db(res_sub.add_parser("completeness"))
    p_re = res_sub.add_parser("export")
    p_re.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_re.add_argument("--out", required=True)
    _add_db(p_re)
    p_ri = res_sub.add_parser("import")
    p_ri.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_ri.add_argument("--path", required=True)
    _add_db(p_ri)

    p_query = sub.add_parser("query", help="Semantic Query (Sprint 9) — read-only")
    query_sub = p_query.add_subparsers(dest="query_cmd", required=True)
    p_qs = query_sub.add_parser("search")
    p_qs.add_argument("--query", required=True, help='Structured query e.g. "domain:DOM_PRICE"')
    _add_db(p_qs)
    p_qi = query_sub.add_parser("inspect")
    p_qi.add_argument("--feature", dest="feature_uuid")
    p_qi.add_argument("--research", dest="research_uuid")
    p_qi.add_argument("--name", dest="canonical_name")
    _add_db(p_qi)
    p_qe = query_sub.add_parser("export")
    p_qe.add_argument("--query", help="Search query (search export)")
    p_qe.add_argument("--feature", dest="feature_uuid")
    p_qe.add_argument("--research", dest="research_uuid")
    p_qe.add_argument("--name", dest="canonical_name")
    p_qe.add_argument("--format", choices=("json", "yaml", "csv"), default="json")
    p_qe.add_argument("--out", required=True)
    _add_db(p_qe)
    p_qv = query_sub.add_parser("validate")
    p_qv.add_argument("--query", required=True)
    _add_db(p_qv)
    _add_db(query_sub.add_parser("capabilities"))
    return parser


def _resolve_db(args: argparse.Namespace) -> Path:
    from feature_intelligence.core.config import load_config

    cfg = load_config()
    db_arg = getattr(args, "db", None)
    if db_arg:
        return Path(os.path.expandvars(str(db_arg))).expanduser()
    return cfg.database.path


def main(argv: list[str] | None = None) -> int:
    from feature_intelligence.core.config import DatabaseConfig, FicConfig, load_config
    from feature_intelligence.core.database import init_database
    from feature_intelligence.core.logging import setup_logging
    from feature_intelligence.migrations.runner import MigrationRunner

    args = _build_parser().parse_args(argv)
    cfg = load_config()
    setup_logging(cfg.logging)
    db_path = _resolve_db(args)
    cfg = FicConfig(
        feature_intelligence=cfg.feature_intelligence,
        database=DatabaseConfig(
            path=db_path,
            timeout_seconds=cfg.database.timeout_seconds,
            journal_mode=cfg.database.journal_mode,
        ),
        logging=cfg.logging,
    )

    if args.command == "init-db":
        print(f"Initialized: {init_database(cfg, apply_migrations=True)}")
        return 0

    if args.command == "migrate":
        runner = MigrationRunner(cfg.database.path)
        if args.downgrade:
            print(f"Rolled back: {runner.downgrade(steps=args.steps) or '(nothing)'}")
        else:
            print(f"Applied: {runner.upgrade() or '(up to date)'}")
        return 0

    if args.command == "status":
        runner = MigrationRunner(cfg.database.path)
        runner.ensure_meta()
        applied = runner.applied_versions()
        print(f"db: {cfg.database.path}")
        print(f"current: {runner.current_version() or '(none)'}")
        print(f"applied: {applied}")
        print(
            "pending: "
            f"{[m.version for m in runner.discover() if m.version not in set(applied)]}"
        )
        return 0

    if args.command == "primitives":
        from feature_intelligence.registry.service import (
            PrimitiveCatalogService,
            PrimitiveNotFoundError,
        )

        svc = PrimitiveCatalogService(cfg.database.path)
        if args.primitives_cmd == "list":
            for row in svc.list_primitives():
                print(
                    f"{row.primitive_id}\t{row.name}\t{row.primitive_type}\t"
                    f"{row.catalog_version}"
                )
            return 0
        if args.primitives_cmd == "get":
            try:
                print(json.dumps(svc.get_primitive(args.primitive_id).__dict__, indent=2))
            except PrimitiveNotFoundError:
                print(f"not found: {args.primitive_id}", file=sys.stderr)
                return 2
            return 0
        if args.primitives_cmd == "validate":
            report = svc.validate_primitives()
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1

    if args.command == "features":
        from dataclasses import asdict

        from feature_intelligence.registry.feature_import_export import (
            export_features,
            import_features,
        )
        from feature_intelligence.registry.feature_service import (
            FeatureNotFoundError,
            FeatureRegistryService,
        )

        svc = FeatureRegistryService(cfg.database.path)
        if args.features_cmd == "list":
            for row in svc.list_features():
                print(
                    f"{row.feature_uuid}\t{row.canonical_name}\t"
                    f"{row.research_state}\t{row.definition_hash[:12]}"
                )
            return 0
        if args.features_cmd == "get":
            try:
                if args.feature_uuid:
                    row = svc.get_by_uuid(args.feature_uuid)
                elif args.canonical_name:
                    row = svc.get_by_name(args.canonical_name)
                else:
                    print("provide --id or --name", file=sys.stderr)
                    return 2
            except FeatureNotFoundError as exc:
                print(f"not found: {exc}", file=sys.stderr)
                return 2
            payload = asdict(row)
            payload["primitive_ids"] = list(row.primitive_ids)
            print(json.dumps(payload, indent=2))
            return 0
        if args.features_cmd == "validate":
            report = svc.validate_registry()
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1
        if args.features_cmd == "export":
            path = export_features(svc, Path(args.out), fmt=args.format)
            print(f"exported: {path}")
            return 0
        if args.features_cmd == "import":
            names = import_features(svc, Path(args.infile), fmt=args.format)
            print(f"imported: {len(names)} features")
            return 0
        if args.features_cmd == "sync":
            data_dir = getattr(args, "data_dir", None)
            chart_dir = getattr(args, "chart_dir", None)
            if not data_dir and chart_dir:
                data_dir = str(Path(chart_dir) / "data")
            if not data_dir:
                # Default: sibling chart/data next to this package
                pkg_chart = Path(__file__).resolve().parent.parent
                candidate = pkg_chart / "data"
                data_dir = str(candidate) if candidate.is_dir() else None
            if not data_dir:
                print(
                    "provide --data-dir or --chart-dir (or run from chart with ./data)",
                    file=sys.stderr,
                )
                return 2
            summary = svc.synchronize_from_feature_registry(
                data_dir,
                mode=getattr(args, "mode", "lenient"),
                force=bool(getattr(args, "force", False)),
                research_sync=bool(getattr(args, "research_sync", True)),
            )
            print(json.dumps(summary.to_dict(), indent=2))
            # Non-zero only when nothing usable was imported and failures exist
            if summary.newly_imported == 0 and summary.already_registered == 0:
                return 1 if summary.failed else 0
            return 0

    if args.command == "operators":
        from dataclasses import asdict

        from feature_intelligence.operators.operator_import_export import (
            export_operators,
            import_operators,
        )
        from feature_intelligence.operators.operator_service import (
            OperatorNotFoundError,
            OperatorRegistryService,
        )

        svc = OperatorRegistryService(cfg.database.path)
        if args.operators_cmd == "list":
            for row in svc.list_operators():
                print(
                    f"{row.operator_id}\t{row.canonical_name}\t"
                    f"{row.category}\t{row.operator_pack_version}"
                )
            return 0
        if args.operators_cmd == "get":
            try:
                if args.operator_id:
                    row = svc.get_by_id(args.operator_id)
                elif args.canonical_name:
                    row = svc.get_by_name(args.canonical_name)
                else:
                    print("provide --id or --name", file=sys.stderr)
                    return 2
            except OperatorNotFoundError as exc:
                print(f"not found: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(asdict(row), indent=2))
            return 0
        if args.operators_cmd == "validate":
            report = svc.validate_registry()
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1
        if args.operators_cmd == "export":
            path = export_operators(svc, Path(args.out), fmt=args.format)
            print(f"exported: {path}")
            return 0
        if args.operators_cmd == "import":
            ids = import_operators(svc, Path(args.infile), fmt=args.format)
            print(f"imported: {len(ids)} operators")
            return 0

    if args.command == "grammar":
        from feature_intelligence.grammar.formatter import format_expression
        from feature_intelligence.grammar.import_export import (
            export_expression,
            import_expression,
        )
        from feature_intelligence.grammar.pack import (
            EXPECTED_GRAMMAR_CHECKSUM,
            FORMATTER_VERSION,
            GRAMMAR_PACK_VERSION,
            GRAMMAR_VERSION,
            TOKEN_PACK_VERSION,
            compute_grammar_pack_checksum,
        )
        from feature_intelligence.grammar.store import GrammarStore
        from feature_intelligence.grammar.validator import validate_text

        def _read_expr() -> str:
            if getattr(args, "expr_text", None):
                return str(args.expr_text)
            if getattr(args, "expr_file", None):
                return Path(args.expr_file).read_text(encoding="utf-8")
            print("provide --file or --expr", file=sys.stderr)
            raise SystemExit(2)

        if args.grammar_cmd == "version":
            checksum = compute_grammar_pack_checksum()
            store = GrammarStore(cfg.database.path)
            row = store.get(GRAMMAR_VERSION) if store.table_exists() else None
            print(f"grammar_version: {GRAMMAR_VERSION}")
            print(f"grammar_pack_version: {GRAMMAR_PACK_VERSION}")
            print(f"token_pack_version: {TOKEN_PACK_VERSION}")
            print(f"formatter_version: {FORMATTER_VERSION}")
            print(f"checksum: {checksum}")
            print(f"expected_checksum: {EXPECTED_GRAMMAR_CHECKSUM}")
            if row is not None:
                print(f"registry_checksum: {row.checksum}")
            return 0
        if args.grammar_cmd == "validate":
            text = _read_expr()
            db = cfg.database.path if args.mode == "bound" else None
            report = validate_text(text, mode=args.mode, db_path=db)
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1
        if args.grammar_cmd == "format":
            try:
                print(format_expression(_read_expr()))
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            return 0
        if args.grammar_cmd == "export":
            path = export_expression(
                _read_expr(), Path(args.out), fmt=args.format
            )
            print(f"exported: {path}")
            return 0
        if args.grammar_cmd == "import":
            canonical = import_expression(
                Path(args.infile), Path(args.out), fmt=args.format
            )
            print(f"imported: {args.out} ({len(canonical)} chars)")
            return 0

    if args.command == "compiler":
        from feature_intelligence.compiler import (
            compile as compile_expr,
            export_transformation,
            import_transformation,
            validate_roundtrip,
        )
        from feature_intelligence.migrations.runner import MigrationRunner

        def _read_expr() -> str:
            if getattr(args, "expr_text", None):
                return str(args.expr_text)
            if getattr(args, "expr_file", None):
                return Path(args.expr_file).read_text(encoding="utf-8")
            print("provide --file or --expr", file=sys.stderr)
            raise SystemExit(2)

        def _bound_db() -> Path:
            """Use --db when provided; otherwise a temporary migrated DB for bound mode."""
            if getattr(args, "db", None):
                return Path(os.path.expandvars(str(args.db))).expanduser()
            import tempfile

            tmp = Path(tempfile.mkdtemp()) / "feature_intelligence.db"
            MigrationRunner(tmp).upgrade()
            return tmp

        persist_db = (
            Path(os.path.expandvars(str(args.db))).expanduser()
            if getattr(args, "db", None)
            else None
        )

        if args.compiler_cmd == "compile":
            text = _read_expr()
            bound_db = _bound_db()
            result = compile_expr(
                text,
                mode="bound",
                db=bound_db,
                persist=persist_db is not None,
                feature_uuid=getattr(args, "feature_uuid", None),
                metrics=bool(getattr(args, "metrics", False)),
                record_cache_hit_event=not bool(
                    getattr(args, "no_cache_hit_event", False)
                ),
            )
            if not result.ok or result.transformation is None:
                print(json.dumps(result.report.to_dict(), indent=2))
                return 1
            print(json.dumps(result.transformation.to_dict(), indent=2))
            return 0

        if args.compiler_cmd == "validate":
            text = _read_expr()
            bound_db = _bound_db()
            if getattr(args, "roundtrip", False):
                report = validate_roundtrip(text, mode="bound", db=bound_db)
            else:
                result = compile_expr(
                    text,
                    mode="bound",
                    db=bound_db,
                    persist=False,
                    record_cache_hit_event=False,
                )
                report = result.report
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1

        if args.compiler_cmd == "manifest":
            text = _read_expr()
            bound_db = _bound_db()
            result = compile_expr(
                text,
                mode="bound",
                db=bound_db,
                persist=False,
                record_cache_hit_event=True,
            )
            if not result.ok or result.transformation is None:
                print(json.dumps(result.report.to_dict(), indent=2))
                return 1
            man = result.transformation.manifest
            print(json.dumps({} if man is None else man.to_dict(), indent=2))
            return 0

        if args.compiler_cmd == "export":
            text = _read_expr()
            bound_db = _bound_db()
            path = export_transformation(
                text,
                Path(args.out),
                fmt=args.format,
                db=bound_db,
                persist=persist_db is not None,
            )
            print(f"exported: {path}")
            return 0

        if args.compiler_cmd == "import":
            bound_db = _bound_db()
            info = import_transformation(
                Path(args.infile),
                fmt=args.format,
                db=bound_db,
                persist=persist_db is not None,
            )
            print(json.dumps(info, indent=2))
            return 0

    if args.command == "ontology":
        from feature_intelligence.ontology.import_export import (
            export_ontology,
            import_ontology,
        )
        from feature_intelligence.ontology.service import (
            OntologyNotFoundError,
            OntologyService,
        )

        svc = OntologyService(cfg.database.path)
        if args.ontology_cmd == "list":
            for row in svc.list_ontology(getattr(args, "object_type", None)):
                print(
                    f"{row.object_type}\t{row.object_id}\t{row.ontology_uuid}\t"
                    f"{row.domain}\t{row.stability}"
                )
            return 0
        if args.ontology_cmd == "get":
            try:
                row = svc.get_ontology(args.object_type, args.object_id)
            except OntologyNotFoundError:
                print(
                    f"not found: {args.object_type}:{args.object_id}",
                    file=sys.stderr,
                )
                return 2
            print(json.dumps(row.to_dict(), indent=2))
            return 0
        if args.ontology_cmd == "validate":
            report = svc.validate_ontology(
                mode=args.mode,
                strict_refs=bool(getattr(args, "strict_refs", False)),
            )
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1
        if args.ontology_cmd == "coverage":
            cov = svc.coverage_ontology()
            print(json.dumps(cov.to_dict(), indent=2))
            return 0
        if args.ontology_cmd == "export":
            path = export_ontology(
                svc,
                Path(args.out),
                fmt=args.format,
                object_type=getattr(args, "object_type", None),
            )
            print(f"exported: {path}")
            return 0
        if args.ontology_cmd == "import":
            report = import_ontology(svc, Path(args.path), fmt=args.format)
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1

    if args.command == "lineage":
        from feature_intelligence.lineage.import_export import (
            export_lineage,
            import_lineage,
        )
        from feature_intelligence.lineage.service import (
            LineageNotFoundError,
            LineageService,
        )

        svc = LineageService(cfg.database.path)
        if args.lineage_cmd == "list":
            for row in svc.list_edges(getattr(args, "relationship_id", None)):
                print(
                    f"{row.lineage_uuid}\t{row.parent_object}\t"
                    f"{row.child_object}\t{row.relationship_id}"
                )
            return 0
        if args.lineage_cmd == "get":
            try:
                if getattr(args, "lineage_uuid", None):
                    row = svc.get_edge(args.lineage_uuid)
                elif (
                    getattr(args, "parent_object", None)
                    and getattr(args, "child_object", None)
                    and getattr(args, "relationship_id", None)
                ):
                    row = svc.get_edge_by_triple(
                        args.parent_object,
                        args.child_object,
                        args.relationship_id,
                    )
                else:
                    print(
                        "require --id LINEAGE_* or "
                        "--parent --child --relationship",
                        file=sys.stderr,
                    )
                    return 2
            except LineageNotFoundError as exc:
                print(f"not found: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(row.to_dict(), indent=2))
            return 0
        if args.lineage_cmd in (
            "parents",
            "children",
            "ancestors",
            "descendants",
        ):
            fn = getattr(svc, args.lineage_cmd)
            for oid in fn(args.object_id):
                print(oid)
            return 0
        if args.lineage_cmd == "validate":
            report = svc.validate_lineage(
                mode=args.mode,
                strict_refs=bool(getattr(args, "strict_refs", False)),
            )
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1
        if args.lineage_cmd == "derive":
            result = svc.derive_lineage(
                transformation_uuid=getattr(args, "transformation_uuid", None),
                feature_uuid=getattr(args, "feature_uuid", None),
                include_closure=not bool(getattr(args, "no_closure", False)),
                strict_refs=bool(getattr(args, "strict_refs", False)),
            )
            print(
                json.dumps(
                    {
                        "upserted": result.upserted,
                        "skipped": result.skipped,
                        "warnings": result.warnings,
                        "edges": len(result.edges),
                    },
                    indent=2,
                )
            )
            return 0
        if args.lineage_cmd == "stats":
            stats = svc.lineage_stats()
            print(json.dumps(stats.to_dict(), indent=2))
            return 0
        if args.lineage_cmd == "export":
            path = export_lineage(
                svc,
                Path(args.out),
                fmt=args.format,
                relationship_id=getattr(args, "relationship_id", None),
            )
            print(f"exported: {path}")
            return 0
        if args.lineage_cmd == "import":
            report = import_lineage(svc, Path(args.path), fmt=args.format)
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1

    if args.command == "research":
        from feature_intelligence.research.import_export import (
            export_research,
            import_research,
        )
        from feature_intelligence.research.service import (
            ResearchNotFoundError,
            ResearchService,
        )

        svc = ResearchService(cfg.database.path)
        if args.research_cmd == "list":
            for row in svc.list_research(getattr(args, "status", None)):
                print(
                    f"{row.research_uuid}\t{row.feature_uuid}\t"
                    f"{row.research_status}\t{row.validation_status}"
                )
            return 0
        if args.research_cmd == "get":
            try:
                if getattr(args, "research_uuid", None):
                    row = svc.get_research(args.research_uuid)
                elif getattr(args, "feature_uuid", None):
                    row = svc.get_research_by_feature(args.feature_uuid)
                else:
                    print("provide --id FRR_* or --feature FEAT_*", file=sys.stderr)
                    return 2
            except ResearchNotFoundError as exc:
                print(f"not found: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(row.to_dict(), indent=2))
            return 0
        if args.research_cmd == "validate":
            report = svc.validate_research(
                mode=args.mode,
                strict_refs=bool(getattr(args, "strict_refs", False)),
                strict_coverage=bool(getattr(args, "strict_coverage", False)),
            )
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1
        if args.research_cmd == "sync":
            summary = svc.sync_research(getattr(args, "feature_uuid", None))
            print(
                f"created={summary.created} updated={summary.updated} "
                f"unchanged={summary.unchanged} skipped={summary.skipped}"
            )
            return 0
        if args.research_cmd == "stats":
            stats = svc.research_stats()
            print(json.dumps(stats.to_dict(), indent=2))
            return 0
        if args.research_cmd == "completeness":
            report = svc.research_completeness()
            print(json.dumps(report.to_dict(), indent=2))
            return 0
        if args.research_cmd == "export":
            path = export_research(svc, Path(args.out), fmt=args.format)
            print(f"exported: {path}")
            return 0
        if args.research_cmd == "import":
            report = import_research(svc, Path(args.path), fmt=args.format)
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1

    if args.command == "query":
        from feature_intelligence.query.service import QueryService

        svc = QueryService(cfg.database.path)
        if args.query_cmd == "search":
            env = svc.search(args.query)
            print(json.dumps(env.to_dict(), indent=2))
            return 0 if env.ok else 1
        if args.query_cmd == "inspect":
            env = svc.inspect(
                feature_uuid=getattr(args, "feature_uuid", None),
                research_uuid=getattr(args, "research_uuid", None),
                canonical_name=getattr(args, "canonical_name", None),
            )
            print(json.dumps(env.to_dict(), indent=2))
            return 0 if env.ok else 1
        if args.query_cmd == "validate":
            env = svc.validate(args.query)
            print(json.dumps(env.to_dict(), indent=2))
            return 0 if env.ok else 1
        if args.query_cmd == "capabilities":
            env = svc.capabilities()
            print(json.dumps(env.to_dict(), indent=2))
            return 0 if env.ok else 1
        if args.query_cmd == "export":
            out = Path(args.out)
            fmt = args.format
            q = getattr(args, "query", None)
            feat = getattr(args, "feature_uuid", None)
            research = getattr(args, "research_uuid", None)
            name = getattr(args, "canonical_name", None)
            try:
                if q:
                    path = svc.export_search(out, query=q, fmt=fmt)
                elif feat or research or name:
                    path = svc.export_inspect(
                        out,
                        feature_uuid=feat,
                        research_uuid=research,
                        canonical_name=name,
                        fmt=fmt,
                    )
                else:
                    print(
                        "export requires --query or --feature/--research/--name",
                        file=sys.stderr,
                    )
                    return 2
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            print(f"exported: {path}")
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
