> 历史快照：2026-10-05 重整前版本；保留原结论与复现记录，不是当前计划。
> 当前入口：[三条 Wave as Prompt 路线](../../wave_as_prompt/README.md)。正文相对链接已重定位。

# Data policy

Large field arrays are never committed. Dataset manifests record canonical
absolute paths, source URLs when public, file hashes, split hashes and the exact
visible/hidden frequency contract. A run must not read a reserved test field
before model and protocol freezing.

## Wave-system representation POC (2026-09-22)

Main 64-grid circle-inclusion systems:

- path: `/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_160.npz`
- bytes: `318420984`
- SHA-256: `589980296257e2e3077312bc39e861aafb162bfec533e7ae0dd302e193e2f80c`
- shape: 160 systems × 9 sources × 8 frequencies × 64 × 64 complex fields;
- split stored in the archive: 112 train / 24 validation / 24 test by system;
- generator config: `configs/wave_operator_representation_poc_20260922.json`.

Unseen ellipse-geometry OOD set:

- path: `/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_64_ood_ellipses_24.npz`
- bytes: `47699264`
- SHA-256: `48a354a3520fbbf0f81b83865b3ffc6afb6f84350a649c59415987e9080fa466`
- generator config: `configs/wave_operator_ood_ellipses_20260922.json`.

Converted legacy 128-grid, three-boundary set:

- path: `/mnt/data/xuangu-fang/physics-informed-tensor-learning/datasets/wave_operator/operator_id_legacy_geometry_128_30.npz`
- bytes: `256071505`
- SHA-256: `485d6a595d4d154ff9db87e71e509744b1cd482ec4ec7e448138c8bb7ef7b860`
- origin: `wavefield-low-rank-representation` `geometry_pool_128_10.npz`;
- conversion entry point: `spectral_continuation.convert_geometry_pool`.
