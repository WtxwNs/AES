# AES experiment matrix

The authoritative machine-readable map is `configs/paper_protocol.json`.
Generate commands and per-cell status with:

```bash
python scripts/paper_matrix.py --horizon 1000000 --manifest paper_matrix.json
```

| Family | Tasks | Carriers | Drift patterns | Variants | Seeds | Cells | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Toy | 1 | 4 | 5 | 2 | 5 | 200 | Runnable portable reconstruction |
| MuJoCo | 5 | 4 | 5 | 2 | 5 | 1000 | Runnable portable reconstruction |
| Isaac | 6 | 4 | 5 | 2 | 5 | 1200 | Blocked integrations remain |

Carriers are SAC, PPO, SQL and the flow-based MEow carrier. SQL is newly
reconstructed, not recovered historical code. Patterns are Steady, Abrupt,
Linear, Periodic and Mixed. Seeds are 1–5. Variants are baseline and AES.

No cell is marked numerically reproduced. The horizon above is a user-selected
reconstruction budget because original per-task horizons were not recovered.
