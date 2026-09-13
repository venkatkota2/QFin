# QFin 1.1.3 follow-up evidence

These JSON records preserve their original provenance and bytes. They are
development/validation observations, not release artifacts or attestations.
The [report](../../hardening-1.1.3.md) explains findings and remaining limits.

| Record | Capture provenance | Scope |
| --- | --- | --- |
| [statistical-workflows.json](statistical-workflows.json) | Clean `061c4ce9fd68ac7487380c8628b88e2cf9a45043`, package 1.1.3 | 80 cells × 100 actual PennyLane executions; minimum coverage 96% |
| [statistical-ambiguity.json](statistical-ambiguity.json) | Parent `f14de6598523ecefc79e925214a926cc720d3e79`, dirty tree, package 1.1.3 | 16 cells × 100 executions; all coverage 100% |
| [paired-alm-reuse.json](paired-alm-reuse.json) | Clean baseline `f14de65`; candidate parent `f14de65`, dirty tree | Four alternating source-verified blocks, ten timings/version/chunk; nine identical array digests |

The ambiguity and performance captures exercised the code subsequently committed
in `061c4ce`; they are not relabelled as clean final-head runs. The paired record's
installed metadata is 1.1.3 in both source checkouts; distinct verified import
paths and source-file hashes identify the actual baseline and candidate.

SHA-256 of the archived raw records:

```text
b4ae83db4ec1d3abda8c980f6a41c513a5dde0ba1078a52ce5cae4901f753423  statistical-workflows.json
74a489134ccedb2eb6f129b8335925b5686058c53994a1f624d14d7aee682979  statistical-ambiguity.json
9aa378f41e32f152d18ca12df135db08b3205062bbe97996d4c9adc616d18cdb  paired-alm-reuse.json
```

The preliminary timing capture that imported the wrong baseline source is not
used here. The corrected harness checks the imported source explicitly. All
performance conclusions use only the verified alternating capture above.
