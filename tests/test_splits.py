"""Checks that the saved split is valid. Run: python -m pytest -q"""
from src.data import load_splits


def test_split_sizes():
    s = load_splits()
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (367, 41, 102)


def test_no_contract_in_two_splits():
    s = load_splits()
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val), "contract in both train and val"
    assert not (train & test), "contract in both train and test"
    assert not (val & test), "contract in both val and test"


def test_no_duplicates_within_a_split():
    for name, titles in load_splits().items():
        assert len(titles) == len(set(titles)), f"duplicate title in {name}"