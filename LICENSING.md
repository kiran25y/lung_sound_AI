# Noise-data licensing policy (commercial product)

Two tiers. **Only the shippable tier may train the model you sell.**

## Shippable tier (commercial-safe)
- **MUSAN — noise category only** — CC BY 4.0. Attribution required. Primary Phase-1 augmentation.
- **Self-recorded noise** — record 30–60 min of ward / clinic / home ambience with your own PCB mic.
  Zero license risk, best spectral match to your hardware. Drop into `data/noise/self_recorded/`.
- **DEMAND** — CC BY-SA (share-alike) — optional extra environments.

## Research tier (experiments / papers ONLY — do NOT ship)
- **DESED_synthetic** — Scaper library is MIT (fine), but the audio is built from FSD50K + SINS + TUT
  with mixed licenses. Use for the Phase-2 **noise-detector** during development and for papers.
  Verify per-source terms before shipping any model trained on it.
- ESC-50, UrbanSound8K, WHAM! — CC BY-NC (non-commercial). Benchmarking only.

The training code reads `noise.tier` from the config and refuses to mix a research-tier corpus when
`noise.tier == "shippable"` (see `phase1_classification/augment/noise_mixer.py`).
