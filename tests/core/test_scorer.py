import logging

import pytest
import regex as re

from prompto.scorer import includes, match, obtain_scoring_functions


def test_obtain_scoring_functions(caplog):
    caplog.set_level(logging.INFO)

    scoring_functions_dict = {"match": match, "includes": includes}

    # raise error if scorer is not a key in scoring_functions_dict
    # scorer is a string case
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Scorer 'not_a_scorer' is not a key in scoring_functions_dict. Available scorers are: ['match', 'includes']"
        ),
    ):
        obtain_scoring_functions(
            scorer="not_a_scorer", scoring_functions_dict=scoring_functions_dict
        )

    # scorer is a list of strings case (of one string)
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Scorer 'not_a_scorer' is not a key in scoring_functions_dict. Available scorers are: ['match', 'includes']"
        ),
    ):
        obtain_scoring_functions(
            scorer=["not_a_scorer"], scoring_functions_dict=scoring_functions_dict
        )

    # scorer is a list of strings case (of multiple strings)
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Scorer 'not_a_scorer' is not a key in scoring_functions_dict. Available scorers are: ['match', 'includes']"
        ),
    ):
        obtain_scoring_functions(
            scorer=["match", "not_a_scorer"],
            scoring_functions_dict=scoring_functions_dict,
        )

    # scorer is a list but some are not strings
    with pytest.raises(
        TypeError, match="If scorer is a list, each element must be a string"
    ):
        obtain_scoring_functions(
            scorer=["match", 1], scoring_functions_dict=scoring_functions_dict
        )

    # passes
    assert obtain_scoring_functions(
        scorer=["match"], scoring_functions_dict=scoring_functions_dict
    ) == [match]
    assert "Scoring functions to be used: ['match']" in caplog.text
    assert obtain_scoring_functions(
        scorer="match", scoring_functions_dict=scoring_functions_dict
    ) == [match]
    assert "Scoring functions to be used: ['match']" in caplog.text
    assert obtain_scoring_functions(
        scorer=["match", "includes"], scoring_functions_dict=scoring_functions_dict
    ) == [match, includes]
    assert "Scoring functions to be used: ['match', 'includes']" in caplog.text
    assert (
        obtain_scoring_functions(
            scorer=[], scoring_functions_dict=scoring_functions_dict
        )
        == []
    )
    assert "Scoring functions to be used: []" in caplog.text


def test_match():
    with pytest.raises(KeyError, match="'response'"):
        match({"expected_response": "hello"})

    with pytest.raises(KeyError, match="'expected_response'"):
        match({"response": "hello"})

    assert match({"response": "hello", "expected_response": "hello"}) == {
        "response": "hello",
        "expected_response": "hello",
        "match": True,
    }
    assert match({"response": "hello", "expected_response": "world"}) == {
        "response": "hello",
        "expected_response": "world",
        "match": False,
    }


def test_includes():
    with pytest.raises(KeyError, match="'response'"):
        includes({"expected_response": "hello"})

    with pytest.raises(KeyError, match="'expected_response'"):
        includes({"response": "hello"})

    assert includes({"response": "hello world", "expected_response": "hello"}) == {
        "response": "hello world",
        "expected_response": "hello",
        "includes": True,
    }
    assert includes({"response": "hello world", "expected_response": "world"}) == {
        "response": "hello world",
        "expected_response": "world",
        "includes": True,
    }
    assert includes({"response": "hello world", "expected_response": "hi"}) == {
        "response": "hello world",
        "expected_response": "hi",
        "includes": False,
    }
    assert includes(
        {"response": "hello world", "expected_response": "hello world!"}
    ) == {
        "response": "hello world",
        "expected_response": "hello world!",
        "includes": False,
    }
