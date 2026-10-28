# External DSA Host runtime integration

Startup, attention, runner and speculative decode hooks delegate to AVSG.
VLLM_ASCEND_AVSG_HOST_PROVIDER=legacy is the default and keeps native behavior.
Use avsg-owned for capability-selected execution. Model-specific names remain
compatibility aliases; legacy omnicache modes remain rollback options.
Drain requests and restart workers to change modes.

## Dependencies

Native base: 73953fc05 (v0.25.1rc source from the frozen image).
Paired AVSG repository: https://gitee.com/arepse/ASVG.git
Paired AVSG commit: 36785ad26f9c4845d8afe03425413c3296960340 (published).
AVSG branch: dsa-physical-pool-primitives
AVSG fingerprint: 0e925f2ce7811497994f2d7662d93ca8556a91d63baa74a2d1ee87ac3782da71

Service admission requires block128. C4 BF16 supports TopK512 and up to eight
queries; uncompressed C8/BF16 supports TopK2048 and up to four queries.
OmniCache uses clean base 4312748 plus standalone physical-pool PR80:
https://gitee.com/zhenhaosun_b078/omni-cache/pulls/80
Pinned commit: 3a0e0cfbd6a5e0d4cf50dd6dd201d2785f89a979.
Old coupled PR71/72/74/75/76/77 are not prerequisites for the owned path.
AVSG operators and matching image libraries remain required.
This is not a deployment bundle.
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

The standalone dependency extraction passed 32 unit tests and 80 selected
AVSG integration tests on A2 CPU (three known fixture-dependent tests were
excluded). Retained primitive definitions are AST-identical to the earlier
implementation. Full model/NPU graph retesting of this clean-backend pairing
is pending. The prior snapshot remains available for rollback.
