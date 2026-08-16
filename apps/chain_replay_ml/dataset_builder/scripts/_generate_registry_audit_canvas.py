"""Generate Registry No-Null audit canvas (writes only under canvases/)."""
from __future__ import annotations

import json
from pathlib import Path

AUDIT = Path(r"D:\data\master_dataset\registry_null_audit_2026-07-24.json")
_canvas_dir = Path(__file__).resolve().parents[4] / "canvases"
_canvas_dir.mkdir(parents=True, exist_ok=True)
OUT = _canvas_dir / "registry-no-null-attribution.canvas.tsx"

d = json.loads(AUDIT.read_text(encoding="utf-8"))
s = d["summary"]
feats = d["features"]
with_null = [f for f in feats if f["null_rows"] > 0]
zero = sum(1 for f in feats if f["null_rows"] == 0)

table_rows = []
for f in with_null:
    if f["status"] in ("bug_suspect", "bug_suspect_or_overlong_warmup"):
        verdict = "BUG"
    elif f["on_nullable_list"]:
        verdict = "Nullable"
    elif f["status"] == "expected_controller_warmup":
        verdict = "Warmup"
    elif f["status"] == "step1_empty_column":
        verdict = "Step1 drop"
    elif f["status"] == "sparse_null_overlap":
        verdict = "Overlap"
    elif f["status"] == "pricing_unavailable":
        verdict = "Pricing"
    else:
        verdict = "OK"
    table_rows.append(
        [
            f["feature"],
            f"{f['null_rows']:,}",
            f"{f['null_pct']}%",
            f"{f['exclusive_rows_removed']:,}",
            f"{f['marginal_rows_saved_if_nullable']:,}",
            "Yes" if f["on_nullable_list"] else "No",
            (
                "Yes"
                if f.get("nullable_candidate") and not f["on_nullable_list"]
                else "No"
            ),
            verdict,
            f["note"][:90],
        ]
    )

bar_cats = []
bar_nulls = []
bar_excl = []
for f in with_null[:12]:
    if f["exclusive_rows_removed"] > 0 or f["null_rows"] >= 20000:
        bar_cats.append(f["feature"][:28])
        bar_nulls.append(int(f["null_rows"]))
        bar_excl.append(int(f["exclusive_rows_removed"]))

parts: list[str] = []
parts.append(
    """import {
  BarChart,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Spacer,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

/**
 * Registry No-Null attribution — Master NIFTY 3s, 2026-07-24.
 * Source: master_dataset_nifty_3s.db (310,002 rows). Pipeline not investigated.
 */
"""
)
parts.append(f"const SUMMARY = {json.dumps(s, indent=2)};\n")
parts.append(f"const ZERO_NULL = {zero};\n")
parts.append(f"const WITH_NULL = {len(with_null)};\n")
parts.append(f"const TABLE_ROWS = {json.dumps(table_rows)};\n")
parts.append(f"const BAR_CATS = {json.dumps(bar_cats)};\n")
parts.append(f"const BAR_NULLS = {json.dumps(bar_nulls)};\n")
parts.append(f"const BAR_EXCL = {json.dumps(bar_excl)};\n")
parts.append(
    """
const BAD_TOKENS = [
  ["63913", "CE", "23450", "7,381"],
  ["63911", "CE", "23400", "7,381"],
  ["63909", "CE", "23350", "7,381"],
  ["63907", "CE", "23300", "7,381"],
  ["63905", "CE", "23250", "5,401"],
  ["63948", "PE", "24200", "4,905"],
  ["63903", "CE", "23200", "3,593"],
  ["63901", "CE", "23150", "2,476"],
  ["63899", "CE", "23100", "699"],
];

export default function RegistryNoNullAttribution() {
  return (
    <Stack gap={24}>
      <Stack gap={8}>
        <H1>Registry No-Null — Why ~60k Extra Rows Vanish</H1>
        <Text tone="secondary">
          Master NIFTY 3s · trading day 2026-07-24 · 206 Registry features ·
          Pipeline not investigated
        </Text>
      </Stack>

      <Callout tone="danger" title="Primary cause: option_low (bug), not 15-minute warmup">
        Of 67,212 rows removed by Registry No-Null, 41,185 are killed exclusively by
        option_low. Nine tokens have option_low NULL for their entire life while
        option_open / option_high / option_prev_close are fully populated. That is the
        day_low latch / stale Master build — do not add option_low to the Nullable
        Feature List.
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value={SUMMARY.raw_total.toLocaleString()} label="Master rows" />
        <Stat value={String(SUMMARY.n_tokens)} label="Tokens (not 20)" />
        <Stat
          value={SUMMARY.warmup_15m_rows.toLocaleString()}
          label="15m warmup rows"
        />
        <Stat
          value={SUMMARY.after_registry_no_null.toLocaleString()}
          label="After Registry No-Null"
          tone="danger"
        />
      </Grid>

      <Card>
        <CardHeader>Funnel vs expectation</CardHeader>
        <CardBody>
          <Table
            headers={["Stage", "Rows", "Delta"]}
            rows={[
              ["Master raw", SUMMARY.raw_total.toLocaleString(), "—"],
              [
                "If only 15m wall-clock warmup",
                SUMMARY.expected_after_warmup_only.toLocaleString(),
                "−" + SUMMARY.warmup_15m_rows.toLocaleString(),
              ],
              [
                "Actual after Registry No-Null",
                SUMMARY.after_registry_no_null.toLocaleString(),
                "−" + SUMMARY.rows_removed_by_no_null.toLocaleString(),
              ],
              [
                "Removed outside 15m warmup",
                SUMMARY.removed_outside_15m_warmup.toLocaleString(),
                "unexpected vs 15m-only model",
              ],
              [
                "After forgiving option_low only",
                "283,975",
                "+41,185 recovered",
              ],
            ]}
          />
          <Spacer height={12} />
          <Text tone="secondary">
            Your ~6,000 warmup estimate assumed 20 contracts. This Master has 50 tokens,
            so a flat 15-minute cut removes 12,600 rows. Even then, ~54.6k removals sit
            outside that window — almost all driven by option_low plus longer EMA
            controller warmups on late-joining tokens.
          </Text>
        </CardBody>
      </Card>

      <H2>Exclusive row killers (unique removals)</H2>
      <Text tone="secondary">
        Exclusive = feature is NULL and every other mandatory Registry feature is non-NULL.
        Only these can independently explain unique row loss.
      </Text>
      <Table
        headers={[
          "Feature",
          "NULL rows",
          "Exclusive removed",
          "If made nullable",
          "On Nullable list?",
          "Verdict",
        ]}
        rows={[
          [
            "option_low",
            "46,598",
            "41,185",
            "+41,185 → 283,975",
            "No — correctly mandatory",
            "BUG — rebuild Master after TDM day_low repair",
          ],
          [
            "iv_ema300",
            "23,470",
            "308",
            "+308",
            "No",
            "Expected long EMA warmup (tiny exclusive)",
          ],
        ]}
        rowTone={["danger", "neutral"]}
      />

      <H2>option_low — full-token NULL (9 / 50)</H2>
      <Grid columns={2} gap={16}>
        <Card>
          <CardHeader>Pattern</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>option_open / high / prev_close: 0 NULLs</Text>
              <Text>option_low: 46,598 NULLs (all day, 9 tokens)</Text>
              <Text>LTP present on every null row</Text>
              <Text>
                Current token_day_meta day_low is valid (positive paise) for these
                tokens — Master was written with NULL lows and has not been rebuilt.
              </Text>
              <Row gap={8}>
                <Pill tone="danger">Not for Nullable list</Pill>
                <Pill tone="warning">Fix upstream / rebuild Master</Pill>
              </Row>
            </Stack>
          </CardBody>
        </Card>
        <Card>
          <CardHeader>Affected tokens</CardHeader>
          <CardBody>
            <Table
              headers={["Token", "Side", "Strike", "Rows (all NULL low)"]}
              rows={BAD_TOKENS}
            />
          </CardBody>
        </Card>
      </Grid>

      <H2>Impact chart (top NULL / exclusive)</H2>
      <BarChart
        categories={BAR_CATS}
        series={[
          { name: "Null rows", data: BAR_NULLS, tone: "warning" },
          { name: "Exclusive removed", data: BAR_EXCL, tone: "danger" },
        ]}
        height={280}
      />
      <Text tone="secondary">
        Source: master_dataset_nifty_3s.db · 2026-07-24 · Registry features only
      </Text>

      <H2>Secondary gap after option_low (~13.5k outside 15m)</H2>
      <Callout tone="info" title="Controller warmup, not Registry bugs">
        After mentally forgiving option_low, 26,027 rows remain incomplete: 12,564 inside
        the first 15 minutes, 13,463 outside. Outside the window the dominant co-NULLs are
        long EMAs (iv_ema300, ltp_ema300, iv_ema200…) on tokens that lack enough prior
        samples — expected readiness, not a second option_low-style bug. Do not put these
        on the Nullable list.
      </Callout>

      <H2>Step 1 empty columns (no row impact)</H2>
      <Row gap={8}>
        <Pill tone="warning">option_vwap — 100% NULL</Pill>
        <Pill tone="warning">futures_vwap — 100% NULL</Pill>
      </Row>
      <Text tone="secondary">
        Dropped in Step 1 before Step 2, so they do not remove rows. Still a data-quality
        bug once VWAPs are required as features (Feature Policy RAW fix + Master rebuild).
      </Text>

      <Divider />

      <H2>
        All Registry features with NULLs ({WITH_NULL} of 206)
      </H2>
      <Text tone="secondary">
        {ZERO_NULL} Registry features have zero NULLs (omitted). Exclusive=0 means the
        feature never uniquely kills a row — its NULLs only appear on rows already
        incomplete for another mandatory reason.
      </Text>
      <Table
        headers={[
          "Feature",
          "NULL rows",
          "%",
          "Exclusive",
          "Marginal if nullable",
          "On list?",
          "Add to list?",
          "Verdict",
          "Note",
        ]}
        rows={TABLE_ROWS}
      />

      <H3>Nullable Feature List (current)</H3>
      <Text>
        gamma_flip_spot, gamma_flip_distance, current_iv, vega, vanna, charm, speed —
        correctly ignored by Step 2. None of these explain the ~60k extra loss.
      </Text>

      <Callout tone="success" title="Next actions (Registry only)">
        1) Confirm token_day_meta day_low &gt; 0 (already true on 2026-07-24 tick DB).
        2) Rebuild Master for that day so option_low fills. 3) Re-run Registry No-Null —
        expect ≈284k–297k depending on EMA policy, not ~243k. 4) Only then investigate
        Pipeline.
      </Callout>
    </Stack>
  );
}
"""
)

OUT.write_text("".join(parts), encoding="utf-8")
print("wrote", OUT, "bytes", OUT.stat().st_size)
