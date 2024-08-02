from unittest.mock import Mock, patch

import pytest


def foo(input_list):
    print(input_list)


def bar(loop_length):
    input_list = []
    for i in range(loop_length):
        input_list.append(i)
        foo(input_list=input_list)


@patch(".foo", new_callable=Mock)
def test_bar(mock_foo):
    bar(4)
    print(mock_foo.call_args_list)
    assert mock_foo.call_count == 4
    mock_foo.assert_called_with(input_list=[0, 1, 2, 3])
    mock_foo.assert_any_call(input_list=[0, 1, 2])
    mock_foo.assert_any_call(input_list=[0, 1])
    mock_foo.assert_any_call(input_list=[0])
