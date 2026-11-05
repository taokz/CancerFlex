import importlib

import pytest

MODULES = [
    "cancerflex", "cancerflex.paths", "cancerflex.synthetic", "cancerflex.infer",
    "cancerflex.matformer", "cancerflex.matformer.granularity", "cancerflex.matformer.nested_mlp",
    "cancerflex.matformer.model",
    "cancerflex.train.dataset", "cancerflex.train.collator", "cancerflex.train.load_model",
    "cancerflex.train.distill_elastic", "cancerflex.train.balanced.weights", "cancerflex.train.balanced.collator",
    "cancerflex.train.balanced.trainer",
    "cancerflex.eval", "cancerflex.eval.u4_axis", "cancerflex.eval.prompt_templates", "cancerflex.eval.prompt_renderer",
    "cancerflex.eval.canonical_image", "cancerflex.eval.eval_iterator", "cancerflex.eval.eval_metrics",
    "cancerflex.eval.bnb_quant", "cancerflex.eval.vision_cache",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_granularity_schedule():
    from cancerflex.matformer.granularity import GranularityController

    c = GranularityController.from_exponential_factor(10240, factor=8)
    assert c.widths == [1280, 2560, 5120, 10240]
    c.set(1280)
    assert c.current_width == 1280
    with pytest.raises(ValueError):
        c.set(20480)
