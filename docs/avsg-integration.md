# External DSA Host runtime integration

Startup, attention, runner and speculative decode hooks delegate to AVSG.
VLLM_ASCEND_AVSG_HOST_PROVIDER=legacy is the default and keeps native behavior.
Use avsg-owned for capability-selected execution. Model-specific names remain
compatibility aliases; legacy omnicache modes remain rollback options.
Drain requests and restart workers to change modes.

## Dependencies

Native base: 73953fc05 (v0.25.1rc source from the frozen image).
Paired AVSG repository: https://gitee.com/arepse/ASVG.git
Paired AVSG commit: aadcf220e428dbdc88e66b6375136a98468d4de5 (published).
AVSG branch: dsa-unified-pre-row-20261028
AVSG fingerprint: c625416be8ccbe846059adf6f59bf7b9beb7a56d9567f8f5d19445a0515e4e20

Service admission requires block128. C4 BF16 supports TopK512 and up to eight
queries; uncompressed C8/BF16 supports TopK2048 and up to four queries.
Frozen OmniCache layout/physical-storage/OX primitives, AVSG operators and
matching image libraries are also required. This is not a deployment bundle.
The tested DSv4 block128 scatter correction is a separate local operator patch
and is intentionally excluded. See avsg-source-lock.json for hashes.
Keep exact source bytes because the external runtime checks native versions.

## Validation scope

- Startup/import isolation: 9 tests passed on A2 without NPU use.
- Paired AVSG consolidation: 209 selected CPU/native/TCP tests passed,
  one HIXL-specific skip; 39 NPU byte/graph component tests passed.
- These are scoped checks, not the full repository suite.
- Full GSM8K/ACS have not been rerun for this final module consolidation;
  previous service results belong to the preceding runtime snapshot.
- The later stable-row-index optimization and its pending validation are
  excluded. No OmniCache product changes or service restarts are included.
