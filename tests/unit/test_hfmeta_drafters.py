"""Speculative-decoding drafters: found on the card or the Hub, turned on only from trusted sources."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2] / "app"))
import hfmeta  # noqa: E402

CARD = ("Pairs with [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash).\n"
        "See github.com/AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4-DFlash\n"
        "```bash\nvllm serve x --speculative-config '{\"method\":\"dflash\",\"model\":\"/drafter\","
        "\"num_speculative_tokens\":11}'\n```\n")


class Api:
    def __init__(self, ids):
        self.ids = ids

    def list_models(self, search, limit):
        return [SimpleNamespace(id=i) for i in self.ids if search.lower() in i.lower()]


def test_card_drafter_replaces_the_authors_local_path():
    assert hfmeta.card_drafters(CARD) == ["z-lab/Qwen3.6-35B-A3B-DFlash"]
    kept, notes = hfmeta.sanitize([("--speculative-config", '{"method":"dflash","model":"/drafter",'
                                                            '"num_speculative_tokens":11}')], None, None, 1,
                                  hfmeta.card_drafters(CARD))
    assert '"model":"z-lab/Qwen3.6-35B-A3B-DFlash"' in kept[0][1] and notes[0][0] == "change"


def test_hub_search_ranks_trusted_dflash_first():
    api = Api(["someone/Qwen3-32B-eagle3", "AngelSlim/Qwen3-32B_eagle3", "z-lab/Qwen3-32B-DFlash", "x/Qwen3-32B-AWQ"])
    found = hfmeta.find_drafters("Qwen/Qwen3-32B", [], "", api)
    assert found[0] == "z-lab/Qwen3-32B-DFlash" and "x/Qwen3-32B-AWQ" not in found
    assert hfmeta.drafter_trusted("AngelSlim/Qwen3-32B_eagle3") and not hfmeta.drafter_trusted("someone/Qwen3-32B-eagle3")
    assert hfmeta.drafter_spec("z-lab/Qwen3-32B-DFlash")["method"] == "dflash"


def test_dynamic_schedule_needs_vllm_033():
    spec = {"method": "dflash", "model": "z/x", "num_speculative_tokens": 11}
    assert "num_speculative_tokens_per_batch_size" not in hfmeta.dynamic_schedule(spec, {"version": "0.32.1"})
    sched = hfmeta.dynamic_schedule(spec, {"version": "0.33.0"})["num_speculative_tokens_per_batch_size"]
    assert sched[0] == [1, 1, 11] and sched[-1][2] <= 2 and all(a[1] < b[0] for a, b in zip(sched, sched[1:]))


def test_spec_decode_off_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("L9S_HOME", str(tmp_path))
    assert not hfmeta.spec_off()
    (tmp_path / "env").write_text("SPEC_DECODE=off\n")
    assert hfmeta.spec_off()


class MlxApi:
    def list_models(self, author, search, limit):
        ids = ["mlx-community/Qwen3-8B-4bit", "mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen3-0.6B-8bit",
               "mlx-community/Qwen3-1.7B-4bit", "mlx-community/Qwen3-VL-2B-4bit"]
        return [SimpleNamespace(id=i) for i in ids if search in i]


def test_mlx_draft_is_the_smallest_same_quant_sibling(monkeypatch):
    monkeypatch.setattr(hfmeta, "_get", lambda url, token=None, timeout=15: '{"vocab_size": 151936}')
    assert hfmeta.find_mlx_draft("mlx-community/Qwen3-8B-4bit", {"vocab_size": 151936}, MlxApi()) == \
        "mlx-community/Qwen3-0.6B-4bit"
    assert hfmeta.find_mlx_draft("mlx-community/Qwen3-0.6B-4bit", {}, MlxApi()) is None
