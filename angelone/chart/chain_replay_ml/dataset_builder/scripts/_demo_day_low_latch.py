"""Demonstrate day_low=0 latch in token_day_meta upsert semantics."""
import sqlite3

sql = """
CREATE TABLE token_day_meta (
  token TEXT, as_of_date TEXT, day_high INT, day_low INT,
  PRIMARY KEY(token, as_of_date)
);
INSERT INTO token_day_meta VALUES ('T','2026-07-24', 41905, 29025);
-- bad tick: day_low=0 (exchange sentinel)
INSERT INTO token_day_meta VALUES ('T','2026-07-24', 41905, 0)
ON CONFLICT(token, as_of_date) DO UPDATE SET
  day_high = MAX(COALESCE(excluded.day_high, token_day_meta.day_high),
                 COALESCE(token_day_meta.day_high, excluded.day_high)),
  day_low  = MIN(COALESCE(excluded.day_low,  token_day_meta.day_low),
                 COALESCE(token_day_meta.day_low,  excluded.day_low));
SELECT day_high, day_low FROM token_day_meta;
-- later good tick with real low 25330 cannot recover
INSERT INTO token_day_meta VALUES ('T','2026-07-24', 41905, 25330)
ON CONFLICT(token, as_of_date) DO UPDATE SET
  day_high = MAX(COALESCE(excluded.day_high, token_day_meta.day_high),
                 COALESCE(token_day_meta.day_high, excluded.day_high)),
  day_low  = MIN(COALESCE(excluded.day_low,  token_day_meta.day_low),
                 COALESCE(token_day_meta.day_low,  excluded.day_low));
SELECT day_high, day_low FROM token_day_meta;
"""
c = sqlite3.connect(":memory:")
for stmt in sql.strip().split(";"):
    s = stmt.strip()
    if not s:
        continue
    cur = c.execute(s)
    if s.upper().startswith("SELECT"):
        print(cur.fetchall())
