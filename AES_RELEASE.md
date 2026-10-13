# AES release guide

The maintained run commands and current status are in [README.md](README.md).
See [REPRODUCTION.md](REPRODUCTION.md) for paper mappings and unresolved inputs,
and [VALIDATION.md](VALIDATION.md) for measured verification.

The legacy Isaac path remains incomplete. Non-steady runs require simulator-backed
setters and must fail when those are absent. The new portable SQL implementation
is in `cleanrl/cleanrl/sql_continuous_action.py`; the external adapter remains
available for integration with the original SQL code when recovered.

Before describing a release as a complete paper replication, all cells in
`configs/paper_protocol.json` must have complete real curves, all missing original
settings must be resolved, and the reported tables must be recomputed from those
curves. Static checks and CPU smoke runs alone are insufficient.
