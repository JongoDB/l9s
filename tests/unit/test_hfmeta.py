"""Model analysis: pure functions, no network."""
import json

import hfmeta

QWEN3_06B = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "num_hidden_layers": 28,
             "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 1024,
             "max_position_embeddings": 40960}
NEMOTRON_H = {"architectures": ["NemotronHForCausalLM"], "num_hidden_layers": 88, "num_attention_heads": 32,
              "num_key_value_heads": 2, "head_dim": 128, "hybrid_override_pattern": "MEMEMEM*E" * 8}
QWEN35_HYBRID = {"text_config": {"num_hidden_layers": 40, "num_attention_heads": 16, "num_key_value_heads": 2,
                                 "head_dim": 256, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 10}}

CAPS = {"version": "0.33.0", "archs": ["Qwen3ForCausalLM"], "quant_methods": ["fp8", "modelopt", "compressed-tensors"],
        "tool_parsers": ["hermes", "qwen3_coder"], "reasoning_parsers": ["qwen3", "nemotron_v3"],
        "flags": {f: {"choices": None, "takes_value": tv} for f, tv in [
            ("--max-model-len", True), ("--kv-cache-dtype", True), ("--trust-remote-code", False),
            ("--tool-call-parser", True), ("--reasoning-parser", True), ("--enable-auto-tool-choice", False),
            ("--tensor-parallel-size", True), ("--speculative-config", True), ("--async-scheduling", False),
            ("--quantization", True)]}}


def test_kv_bytes_full_attention_matches_vllm():
    # vLLM measured ~114.7 KB/token for Qwen3-0.6B (bf16 KV)
    lay = hfmeta.attention_layout(QWEN3_06B)
    assert lay["attn_layers"] == 28
    assert hfmeta.kv_bytes_per_token(QWEN3_06B, lay, 2) == 2 * 28 * 8 * 128 * 2


def test_hybrid_layouts_only_count_attention_layers():
    lay = hfmeta.attention_layout(NEMOTRON_H)
    assert lay["attn_layers"] == 8 and lay["linear_layers"] == 32
    tc = hfmeta._text_cfg(QWEN35_HYBRID)
    lay = hfmeta.attention_layout(tc)
    assert lay["attn_layers"] == 10 and lay["linear_layers"] == 30


def test_classify_formats_per_backend():
    assert hfmeta.classify_format("unsloth/Qwen3-0.6B-GGUF", ["gguf"], None) == ("GGUF", False)
    assert hfmeta.classify_format("mlx-community/Qwen3-8B-4bit", ["mlx"], None)[1] is False
    assert hfmeta.classify_format("mlx-community/Qwen3-8B-4bit", ["mlx"], None, "mlx") == ("MLX", True)
    assert hfmeta.classify_format("Qwen/Qwen3-8B-FP8", [], {"quant_method": "fp8"})[0] == "FP8"
    assert hfmeta.classify_format("Qwen/Qwen3-8B-FP8", [], {"quant_method": "fp8"}, "mlx")[1] is False


def test_download_plan_skips_unused_files():
    files = [("model.safetensors", 100), ("pytorch_model.bin", 100), ("original/consolidated.pth", 500),
             ("model.gguf", 50), ("config.json", 1)]
    plan = hfmeta.download_plan(files)
    assert plan["download_bytes"] == 101
    import fnmatch
    excluded = [f for f, _ in files if any(fnmatch.fnmatch(f, p) for p in plan["exclude"])]
    assert sorted(excluded) == ["model.gguf", "original/consolidated.pth", "pytorch_model.bin"]


def test_sanitize_removes_what_this_server_cant_take():
    pairs = [("--host", "0.0.0.0"), ("--swap-space", "16"), ("--max-model-len", "1000000"),
             ("--tensor-parallel-size", "8"), ("--reasoning-parser", "super_v3"),
             ("--reasoning-parser-plugin", "/app/x.py"), ("--kv-cache-dtype", "fp8"), ("--trust-remote-code", None)]
    kept, notes = hfmeta.sanitize(pairs, CAPS, fit_ctx=262144, tp=1)
    assert kept == [("--max-model-len", "262144"), ("--kv-cache-dtype", "fp8"), ("--trust-remote-code", None)]
    kept, _ = hfmeta.sanitize([("--tensor-parallel-size", "8")], CAPS, fit_ctx=None, tp=2)
    assert kept == [("--tensor-parallel-size", "2")]


def test_card_command_extraction_prefers_hardware_section():
    card = """# Model
## vLLM
```bash
vllm serve org/model --max-model-len 8192 --gpu-memory-utilization 0.9
```
##### vLLM on DGX Spark
```bash
docker run --gpus all -e VLLM_X=1 vllm/vllm-openai:v0.30.0 --model org/model --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 --speculative_config '{"method":"mtp","num_speculative_tokens":3,"model":"org/drafter"}'
```
## SGLang
```bash
python3 -m sglang.launch_server --model org/model
```
"""
    cmds = [hfmeta.parse_command(c, CAPS) for c in hfmeta.extract_card_commands(card)]
    assert len(cmds) == 2  # sglang ignored
    for c in cmds:
        c["kept"], c["notes"] = hfmeta.sanitize(c["pairs"], CAPS, 65536)
        c["score"] = hfmeta.score_command(c, {"spark", "gb10", "dgx spark"})
    best = max(cmds, key=lambda c: c["score"])
    assert best["heading"] == "vLLM on DGX Spark"
    assert best["env"] == {"VLLM_X": "1"}
    assert ("--speculative-config", '{"method":"mtp","num_speculative_tokens":3,"model":"org/drafter"}') in best["kept"]
    assert hfmeta.spec_drafter({f: (v, "") for f, v in best["kept"]}) == "org/drafter"


def test_validate_flags_reports_unknown_flags_and_parsers():
    probs = hfmeta.validate_flags("--bogus 1 --tool-call-parser nope --max-model-len 4096", CAPS)
    assert any("--bogus" in p for p in probs) and any("nope" in p for p in probs)
    assert hfmeta.validate_flags("--max-model-len 4096 --tool-call-parser hermes", CAPS) == []


def test_analyze_end_to_end_offline():
    meta = {"repo": "Qwen/Qwen3-0.6B", "files": [("model.safetensors", 1_500_000_000), ("config.json", 1)],
            "tags": ["safetensors"], "config": QWEN3_06B, "generation_config": {"temperature": 0.6},
            "chat_template": "{% if tools %}<tool_call>{% endif %}<think>", "readme": ""}
    A = hfmeta.analyze(meta, CAPS, mem_total=128e9, gpu_mem=0.85)
    assert A["verdict"] == "ok" and A["arch_supported"] is True
    assert A["ctx"] == 40960 and A["parsers"]["tool"] == "hermes" and A["parsers"]["reasoning"] == "qwen3"
    flags = A["presets"][0]["flags"]
    assert flags["--max-model-len"][0] == "40960" and flags["--tool-call-parser"][0] == "hermes"
    json.dumps(hfmeta.render_markdown(A, CAPS))  # renders without errors
