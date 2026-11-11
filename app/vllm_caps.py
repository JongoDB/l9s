"""Runs INSIDE the vLLM container. Prints a JSON description of what this vLLM build supports:
serve flags (+ allowed choices), model architectures, tool/reasoning parsers, quantization methods."""
import json
import sys

out = {}
try:
    import vllm
    out["version"] = vllm.__version__
except Exception as e:  # noqa: BLE001
    out["version"] = f"? ({e})"

try:
    from vllm.entrypoints.openai.cli_args import make_arg_parser
except Exception:  # newer layout
    from vllm.entrypoints.common.cli_args import make_arg_parser
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
except Exception:
    from vllm.utils import FlexibleArgumentParser

parser = make_arg_parser(FlexibleArgumentParser())
flags = {}
for act in parser._actions:
    for opt in act.option_strings:
        if opt.startswith("--"):
            flags[opt] = {
                "choices": [str(c) for c in act.choices] if act.choices else None,
                "takes_value": act.nargs != 0,
            }
out["flags"] = flags

from vllm import ModelRegistry  # noqa: E402
out["archs"] = sorted(ModelRegistry.get_supported_archs())


def names(mgr, attr_list, attr_dict):
    if hasattr(mgr, attr_list):
        return sorted(getattr(mgr, attr_list)())
    return sorted(getattr(mgr, attr_dict).keys())


try:
    from vllm.tool_parsers import ToolParserManager
except Exception:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
out["tool_parsers"] = names(ToolParserManager, "list_registered", "tool_parsers")
try:
    from vllm.reasoning import ReasoningParserManager
    out["reasoning_parsers"] = names(ReasoningParserManager, "list_registered", "reasoning_parsers")
except Exception:
    out["reasoning_parsers"] = []
try:
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    out["quant_methods"] = sorted(QUANTIZATION_METHODS)
except Exception:
    out["quant_methods"] = []
json.dump(out, sys.stdout)
