# Correlation Engine Benchmark (Phase 1)

- Generated: `2026-07-31T09:19:40.879002+00:00`
- Platform: `nt`
- GPU available: **False**
- Features: `20`
- Preference: `auto`

| n_rows | n_features | CPU (s) | GPU transfer (s) | GPU compute (s) | GPU total (s) | Speedup | Memory (MB) | Backend | Status |
|-------:|----------:|--------:|-----------------:|----------------:|--------------:|--------:|------------:|---------|--------|
| 100,000 | 20 | 0.0475 | N/A | N/A | N/A | N/A | 94.0742 | cpu | gpu_na |
| 500,000 | 20 | 0.2402 | N/A | N/A | N/A | N/A | 155.1211 | cpu | gpu_na |
| 1,000,000 | 20 | 0.5706 | N/A | N/A | N/A | N/A | 231.4219 | cpu | gpu_na |
| 5,000,000 | 20 | 2.4599 | N/A | N/A | N/A | N/A | 841.7930 | cpu | gpu_na |
| 10,000,000 | 20 | 5.1137 | N/A | N/A | N/A | N/A | 1603.9883 | cpu | gpu_na |

## Notes
- CPU remains the default Analysis Lab backend.
- RAPIDS cuDF is typically Linux+CUDA only; Windows runs CPU-only.
- Speedup = cpu_time_sec / gpu_total_sec when GPU succeeds.
- n_rows=100000: GPU N/A — RAPIDS cuDF/CuPy not importable (common on Windows; use Linux+CUDA for GPU path)
