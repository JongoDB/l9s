# Changelog

## 0.2.2
- Hub: all columns fit beside the detail panel (long repo ids shortened, org first; relative "updated" ages; ✓ column for downloaded models).
- Fresh screenshots.

## 0.2.1
- Fix: clicking Settings (header `<7>`, breadcrumb) from the Requests or Local view snapped back to the previous view.
- Arrow keys: `←`/`→` switch views (and move between buttons in dialogs); `↑`/`↓` move past the end of a table into the next one in the view (Requests ↔ Clients, Profiles ↔ Models, Hub search ↔ results) and between fields in Settings.
- Tables spread their columns across the full width, k9s-style, instead of packing them to the left.

## 0.2.0
- Python dependencies (incl. mlx-lm and MLX on a Mac) are pinned in `requirements.lock`; installs and updates use exactly the CI-tested set.
- CI on every push: Linux unit tests and installer, plus an Apple Silicon job that serves a real model with mlx-lm behind the proxy.
- Weekly dependency refresh: newer versions are tested in CI and, if green, released automatically as a patch version with a changelog entry; failures open an issue. New vLLM releases open an issue for manual testing.
- `get.sh` and `l9s update` install the latest GitHub release instead of `main`.

## 0.1.0
- First release: k9s-style TUI and OpenAI-compatible endpoint for local LLMs.
- Deployments: vLLM in Docker on Linux GPUs (tuned for GB10 / DGX Spark), mlx-lm on Apple Silicon.
- Request proxy: per-request view, cancel, events, API key.
- Model analysis from Hugging Face: fit, context, parsers, model-card commands checked against your server.
- Smart downloads, one-line installer, `l9s quickstart`, `l9s uninstall`.
- Update checks: `l9s update`, daily background check, notify or auto-update.
