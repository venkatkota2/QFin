# Current QFin documentation

These pages describe the current source and installed package contract. Obtain the
installed version with `qfin.__version__` or `qfin --version`; package metadata is
the version source. The hardening milestone changes no modelling scope.

| Topic | Document |
| --- | --- |
| Responsibilities and execution boundaries | [Architecture](architecture.md) |
| Dates, settlement, accrual, units | [Financial conventions](financial-conventions.md) |
| Curve interpolation, scenarios, bootstrapping | [Curves](curves.md) |
| Pricing, yields, duration and key rates | [Fixed income](fixed-income.md) |
| Asset-liability valuation and scenarios | [ALM](alm.md) |
| Annual policy projection | [Life](life.md) |
| Weighted finite loss statistics | [Risk](risk.md) |
| Compilation decisions and error units | [Compiler](compiler.md) |
| Distribution and objective encodings | [Representation](representation.md) |
| Experimental quantum risk | [Quantum risk](quantum-risk.md) |
| Dispatch policy and measurements | [Performance](performance.md) |
| Numerical methods and tolerances | [Numerical methodology](numerical-methodology.md) |
| Tests, packaging and CI | [Validation](validation.md) |
| Stable names and deprecations | [Public API](public-api.md) |
| Explicit boundaries | [Limitations](limitations.md) |
| 1.1.2 findings and A–L evidence | [Hardening report](hardening-1.1.2.md) |
| 1.1.3 remaining-issue fixes | [Follow-up report](hardening-1.1.3.md) |
| Seeds, versions and execution provenance | [Reproducibility](reproducibility.md) |
| Workflow sampling bounds and empirical coverage | [Statistical validation](statistical-validation.md) |
| Chunking, allocation guards and peak RSS | [Memory](memory.md) |
| Compatible failure categories | [Exceptions](exceptions.md) |
| Exact tested wheels, sdist and provenance | [Release engineering](release-engineering.md) |
| Required repository administration | [Repository settings](repository-settings.md) |

[Historical documents](history/README.md) retain earlier designs and benchmark
evidence. Historical timings are not current dispatch recommendations.
