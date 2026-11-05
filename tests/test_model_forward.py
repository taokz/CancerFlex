import torch
from transformers import Gemma3ForConditionalGeneration

from cancerflex.matformer.granularity import GranularityController
from cancerflex.matformer.model import wrap_medgemma_mlps
from cancerflex.matformer.nested_mlp import NestedGemma3MLP

from conftest import tiny_config


def _text_forward(model, ids):
    return model(input_ids=ids).logits


def test_wrap_and_full_width_equivalence():
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(tiny_config()).eval()
    ids = torch.randint(7, 64, (1, 6))
    with torch.no_grad():
        ref = _text_forward(model, ids)
    controller = GranularityController(widths=[8, 16, 32, 64])
    n = wrap_medgemma_mlps(model, controller)
    assert n == 2  # one MLP per decoder layer
    assert any(isinstance(m, NestedGemma3MLP) for m in model.modules())
    with torch.no_grad():
        full = _text_forward(model, ids)
    torch.testing.assert_close(full, ref)


def test_narrow_width_runs_and_differs():
    torch.manual_seed(0)
    model = Gemma3ForConditionalGeneration(tiny_config()).eval()
    controller = GranularityController(widths=[8, 16, 32, 64])
    wrap_medgemma_mlps(model, controller)
    ids = torch.randint(7, 64, (2, 5))
    with torch.no_grad():
        full = _text_forward(model, ids)
        controller.set(8)
        narrow = _text_forward(model, ids)
    assert narrow.shape == full.shape
    assert torch.isfinite(narrow).all()
    assert not torch.allclose(narrow, full)


def test_multimodal_forward_with_synthetic_image(tiny_checkpoint):
    from cancerflex.infer import load_model
    from cancerflex.synthetic import make_synthetic_image

    model, processor, controller = load_model(tiny_checkpoint, device="cpu", dtype="fp32")
    assert controller.widths == [8, 16, 32, 64]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Is a lesion visible ?"}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, images=[make_synthetic_image(size=64)], return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    assert out.logits.shape[0] == 1 and torch.isfinite(out.logits).all()
