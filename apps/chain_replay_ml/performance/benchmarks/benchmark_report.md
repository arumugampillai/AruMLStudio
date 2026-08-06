# Feature Engine Performance Benchmark (Phase 6.0)

Generated: `2026-07-31T06:42:13.963541+00:00`

- rows: **100000**
- features_assumed: **206**
- numba_available: **True**
- overall suite speedup: **3.738x**

## Limitations

- Synthetic price/IV streams — not a full production day build.
- Does not exercise build_feature_raw_for_row end-to-end (requires tick DB).
- Throughput approximates controller hotspots inside the per-row feature path.
- features/sec uses assumed feature count=206 (registry-sized).

## Compile overhead (first call)

- `population_std`: 0.3240s
- `ema_update`: 0.0021s
- `ema_series`: 0.0039s
- `rolling_mean_std`: 0.0038s
- `rolling_max_min`: 0.0037s
- `pct_returns`: 0.0028s
- `safe_ratio`: 0.0019s
- `distance_pct`: 0.0019s
- `iv_zscore`: 0.0024s

## Per-kernel speedup

| kernel | baseline_sec | numba_sec | speedup |
|---|---:|---:|---:|
| `StdController_vs_legacy_list_np` | 1.4782 | 0.3789 | 3.901x |
| `population_std` | 1.3059 | 0.0538 | 24.272x |
| `ema_series` | 0.0273 | 0.0001 | 192.006x |
| `iv_zscore` | 1.6189 | 0.0852 | 18.992x |
| `StdController` | 1.7840 | 0.3642 | 4.899x |
| `RvController` | 1.8569 | 0.3977 | 4.669x |
| `IvZscoreWindowController` | 2.1267 | 1.0202 | 2.085x |

## Suite profiles

### controller_suite_baseline

- total_sec: **33.0754**
- rows: 100000 → **3,023 rows/sec**
- features: 206 → **622,819 features/sec**
- peak_memory_mib: 0.781

| function | file | cumtime | tottime | calls |
|---|---|---:|---:|---:|
| `<lambda>` | benchmark.py:306 | 33.0754 | 0.0000 | 1 |
| `_suite` | benchmark.py:298 | 33.0754 | 0.0000 | 1 |
| `update` | rolling_controllers.py:174 | 31.8610 | 0.7017 | 300000 |
| `_population_std_buffer` | rolling_controllers.py:257 | 20.9778 | 0.1897 | 199951 |
| `population_std` | runtime.py:73 | 20.7881 | 0.1875 | 199951 |
| `population_std_numpy` | feature_kernels.py:47 | 19.1135 | 0.2502 | 199951 |
| `std` | fromnumeric.py:3913 | 18.8324 | 0.4666 | 199951 |
| `_std` | _methods.py:225 | 18.3658 | 1.1680 | 199951 |
| `_var` | _methods.py:153 | 17.1382 | 9.9424 | 199951 |
| `_bench_rv_controller` | benchmark.py:164 | 12.5901 | 0.1790 | 1 |
| `_update_sample` | rolling_controllers.py:318 | 11.9880 | 0.5788 | 100000 |
| `_bench_std_controller` | benchmark.py:140 | 11.8105 | 0.1744 | 1 |

### controller_suite_numba

- total_sec: **8.8485**
- rows: 100000 → **11,301 rows/sec**
- features: 206 → **2,328,083 features/sec**
- peak_memory_mib: 0.774

| function | file | cumtime | tottime | calls |
|---|---|---:|---:|---:|
| `<lambda>` | benchmark.py:314 | 8.8485 | 0.0000 | 1 |
| `_suite` | benchmark.py:298 | 8.8485 | 0.0000 | 1 |
| `update` | rolling_controllers.py:174 | 8.1059 | 0.6209 | 300000 |
| `_bench_iv_zscore_controller` | benchmark.py:188 | 4.0343 | 0.2271 | 1 |
| `_update_sample` | rolling_controllers.py:368 | 3.4688 | 1.6655 | 100000 |
| `_bench_rv_controller` | benchmark.py:164 | 2.6143 | 0.1531 | 1 |
| `_bench_std_controller` | benchmark.py:140 | 2.1997 | 0.1543 | 1 |
| `_update_sample` | rolling_controllers.py:318 | 2.1239 | 0.5232 | 100000 |
| `buffer_to_float64` | feature_kernels.py:233 | 2.0671 | 0.5042 | 299853 |
| `_population_std_buffer` | rolling_controllers.py:257 | 1.8034 | 0.1398 | 199951 |
| `_update_sample` | rolling_controllers.py:278 | 1.7079 | 0.2660 | 100000 |
| `population_std` | runtime.py:73 | 1.6636 | 0.2828 | 199951 |

## How to re-run

```bash
cd angelone/chart
python -m chain_replay_ml.performance.benchmark --rows 100000
```

Disable Numba: `set ARUNEO_FEATURE_NUMBA=off` (Windows) / `export ARUNEO_FEATURE_NUMBA=off`.
