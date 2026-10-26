"""Saving answers: empty answers are not saved, and a dict counts only when every entry is filled."""
import habtrain as ht


def test_filled():
    assert not ht.record.filled(...)
    assert not ht.record.filled("")
    assert not ht.record.filled({"a": 1, "b": ...})
    assert ht.record.filled({"a": 1, "b": 2})
    assert ht.record.filled(0)
