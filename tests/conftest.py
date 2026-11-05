"""Shared fixtures: a tiny, randomly initialised Gemma-3 style multimodal model saved to disk.

No real weights are used anywhere in the test-suite; everything runs on CPU in seconds.
"""
from __future__ import annotations

import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3Processor,
    PreTrainedTokenizerFast,
    Gemma3ImageProcessor,
)

IMAGE_SIZE = 32
PATCH = 16
MM_TOKENS = 4  # (32/16)^2 patches pooled to 4 tokens


def tiny_config() -> Gemma3Config:
    text = dict(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=256, sliding_window=16,
        hidden_activation="gelu_pytorch_tanh",
    )
    vision = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        image_size=IMAGE_SIZE, patch_size=PATCH, num_channels=3, vision_use_head=False,
    )
    cfg = Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=MM_TOKENS,
                       image_token_index=4, boi_token_index=5, eoi_token_index=6)
    cfg.text_config.vocab_size = 64
    return cfg


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<image_soft_token>": 4,
             "<start_of_image>": 5, "<end_of_image>": 6, "<start_of_turn>": 7, "<end_of_turn>": 8}
    words = ["user", "model", "Is", "a", "lesion", "visible", "in", "this", "image", "?", "Answer", "Yes", "or",
             "No", ".", "\n", "the", "and", "of", "to"]
    for w in words:
        vocab.setdefault(w, len(vocab))
    while len(vocab) < 64:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>",
        additional_special_tokens=["<image_soft_token>", "<start_of_image>", "<end_of_image>",
                                   "<start_of_turn>", "<end_of_turn>"],
        extra_special_tokens={"image_token": "<image_soft_token>", "boi_token": "<start_of_image>",
                              "eoi_token": "<end_of_image>"},
    )
    t.chat_template = (
        "{{ bos_token }}{% for m in messages %}<start_of_turn> {{ m['role'] }} \n"
        "{% for c in m['content'] %}{% if c['type'] == 'image' %}<start_of_image> {% elif c['type'] == 'text' %}{{ c['text'] }} {% endif %}{% endfor %}"
        "<end_of_turn> \n{% endfor %}{% if add_generation_prompt %}<start_of_turn> model \n{% endif %}"
    )
    return t


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    """Save a random tiny Gemma-3 multimodal model + processor + granularities file; return the directory."""
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny_ckpt")
    cfg = tiny_config()
    model = Gemma3ForConditionalGeneration(cfg)
    model.save_pretrained(d)
    image_processor = Gemma3ImageProcessor(size={"height": IMAGE_SIZE, "width": IMAGE_SIZE}, do_convert_rgb=True, do_pan_and_scan=False)
    processor = Gemma3Processor(image_processor=image_processor, tokenizer=tiny_tokenizer(),
                                image_seq_length=MM_TOKENS)
    processor.save_pretrained(d)
    (d / "matformer_granularities.json").write_text(json.dumps({"widths": [8, 16, 32, 64]}))
    return d
