"""Windowing checks on a small synthetic contract. Run: python -m pytest -q"""
import pytest
from transformers import AutoTokenizer

from src.preprocessing import char_span_to_tokens, make_windows, tokens_to_char_span

QUESTION = 'Highlight the parts (if any) of this contract related to "Governing Law" that should be reviewed by a lawyer.'
ANSWER = "This Agreement shall be governed by the laws of the State of New York."
FILLER = "The parties agree to the terms set out in this section.  \n\n"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base")


def make_contract():
    """About 3,000 tokens of filler with the answer in the middle (messy whitespace on purpose)."""
    before = FILLER * 120
    return before + ANSWER + "\n\n" + FILLER * 120, len(before)


def test_windows_fit_and_overlap(tokenizer):
    contract, _ = make_contract()
    windows = make_windows(tokenizer, QUESTION, contract, max_length=512, stride=128)
    assert len(windows) > 1
    assert all(len(w["input_ids"]) <= 512 for w in windows)
    for a, b in zip(windows, windows[1:]):
        a_end = max(o[1] for o in a["offsets"] if o)
        b_start = min(o[0] for o in b["offsets"] if o)
        assert b_start < a_end, "consecutive windows must overlap"


def test_answer_round_trip(tokenizer):
    contract, start = make_contract()
    found = 0
    for w in make_windows(tokenizer, QUESTION, contract, max_length=512, stride=128):
        tok = char_span_to_tokens(w["offsets"], start, start + len(ANSWER))
        if tok:
            cs, ce = tokens_to_char_span(w["offsets"], *tok)
            assert contract[cs:ce] == ANSWER
            found += 1
    assert found >= 1, "the answer must lie fully inside at least one window"


def test_answer_outside_window_is_no_answer(tokenizer):
    contract, start = make_contract()
    first = make_windows(tokenizer, QUESTION, contract, max_length=512, stride=128)[0]
    assert char_span_to_tokens(first["offsets"], start, start + len(ANSWER)) is None
