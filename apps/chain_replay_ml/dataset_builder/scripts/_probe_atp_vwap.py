import sqlite3
from pathlib import Path

tick_dir = Path(r"D:/data/ticks")
path = tick_dir / "angel_market_2026-07-24.db"
if not path.exists():
    cands = sorted(
        [p for p in tick_dir.glob("angel_market_*.db") if p.stat().st_size > 1_000_000],
        key=lambda p: p.name,
    )
    path = cands[-1]
print("using", path)

conn = sqlite3.connect(str(path))
cols = {r[1] for r in conn.execute("PRAGMA table_info(ticks)")}
print("has atp", "atp" in cols)

row = conn.execute(
    """
    SELECT COUNT(*) n,
           SUM(CASE WHEN atp IS NULL THEN 1 ELSE 0 END) null_atp,
           SUM(CASE WHEN atp = 0 THEN 1 ELSE 0 END) zero_atp,
           SUM(CASE WHEN atp > 0 THEN 1 ELSE 0 END) pos_atp,
           MIN(CASE WHEN atp > 0 THEN atp END) min_pos,
           MAX(atp) max_atp
    FROM ticks
    """
).fetchone()
print("ticks atp:", dict(zip(["n", "null", "zero", "pos", "min_pos", "max"], row)))
print("sample:", conn.execute("SELECT token, ltp, atp FROM ticks WHERE ltp>0 LIMIT 5").fetchall())
print("pos sample:", conn.execute("SELECT token, ltp, atp FROM ticks WHERE atp>0 LIMIT 5").fetchall())

tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
print("tables", sorted(tables))
if "token_day_meta" in tables:
    for itype in ("OPTIDX", "FUTIDX", "OPTSTK", "FUTSTK"):
        tot = conn.execute(
            """
            SELECT COUNT(*) FROM ticks t
            JOIN token_day_meta m ON m.token=t.token
            WHERE m.instrument_type=?
            """,
            (itype,),
        ).fetchone()[0]
        pos = conn.execute(
            """
            SELECT COUNT(*) FROM ticks t
            JOIN token_day_meta m ON m.token=t.token
            WHERE m.instrument_type=? AND t.atp>0
            """,
            (itype,),
        ).fetchone()[0]
        print(f"{itype}: ticks={tot:,} atp>0={pos:,}")
conn.close()

mconn = sqlite3.connect(r"D:/data/master_dataset/master_dataset_nifty_3s.db")
mcols = {r[1] for r in mconn.execute("PRAGMA table_info(samples)")}
days = [r[0] for r in mconn.execute("SELECT DISTINCT trading_day FROM samples ORDER BY 1 DESC LIMIT 3")]
print("master days", days)
day = days[0] if days else None
if day:
    for c in ("option_vwap", "futures_vwap", "futures_ltp", "ltp"):
        if c not in mcols:
            print(c, "MISSING")
            continue
        n, nulls, pos = mconn.execute(
            f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN "{c}" IS NOT NULL AND "{c}" > 0 THEN 1 ELSE 0 END)
            FROM samples WHERE trading_day=?
            """,
            (day,),
        ).fetchone()
        print(f"master {day} {c}: n={n:,} null={nulls:,} pos={pos:,}")
mconn.close()
