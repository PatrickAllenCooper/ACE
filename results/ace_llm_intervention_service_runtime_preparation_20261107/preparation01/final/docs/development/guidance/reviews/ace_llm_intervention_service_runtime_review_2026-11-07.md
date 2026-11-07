# Prospective service runtime review

Date: November 7, 2026. Independent reviewer: Mendel, a read-only agent separate from the implementation parent. This is a code review, not an actual runtime or scientific qualification.

Scope: the prospective v2 model service, its new runtime adapter, the changed authentication fixtures, new runtime fixtures and revised service guide. The reviewer compared the adapter with the unchanged accepted CPU diagnostic and its terminal/acceptance evidence. No tests, remote commands, model imports or archive hashing were performed by the reviewer.

The final review found no required fixes. One observation was closed during review. Four additional imported package versions were validated but initially absent from the saved runtime snapshot. The implementation now records those values before validation, including missing values as null. This change does not relax acceptance.

The reviewer confirmed the separation of imported versions and distribution metadata, rejection of historical-runtime relabeling, exact reuse of generated-Torch provenance checks, filtering of authenticated captured project objects, explicit rejection of unverified aliases, private temporary roots and isolated execution. CPU verification hides CUDA before project imports and does not load weights or deserialize adapters. The model recipe and five-field handshake are unchanged.

The stage receipt binds metadata supplied by a trusted external archive stager. It is not proof that staging occurred. Actual stager execution, supervised CPU preflight, subsequent GPU/model-service qualification and scientific admission remain open.

Reviewed runtime adapter SHA-256: `27b40bd3cf8769cef2e345dc370591112b5519be9c7717eb15fc09a06974a25d`.

The optional suggestion to test preservation of an extra-package version mismatch was implemented as one additional artificial method. It passed once and retained the mismatching value in saved evidence before rejection. Earlier methods were not repeated. This follow-up is a parent implementation/test disposition, not a claim that the independent reviewer executed it.
