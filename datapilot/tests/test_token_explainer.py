from scripts.token_explainer import BUILT_IN_CORPUS, build_vocab, encode, train


def merges():
    return train(BUILT_IN_CORPUS, 12, show=False)


def test_common_word_becomes_one_token():
    assert encode("low", merges(), show=False) == ["low"]


def test_rarer_word_splits_into_more_tokens():
    m = merges()
    assert len(encode("lowest", m, show=False)) > len(encode("low", m, show=False))


def test_pieces_rebuild_the_word_and_have_ids():
    m = merges()
    vocab = build_vocab(m)
    pieces = encode("payments", m, show=False)
    assert "".join(pieces) == "payments"
    assert all(p in vocab for p in pieces)
