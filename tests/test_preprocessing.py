import pytest
import pandas as pd
import numpy as np

from src.preprocessing import (
    validate_dataframe,
    clean_text_columns,
    handle_missing_values,
    encode_categorical_variables
)


def test_handle_missing_values_numeric():
    df = pd.DataFrame({
        "age": [50, np.nan, 60],
        "chol": [200, 240, np.nan]
    })

    result = handle_missing_values(df)

    assert result.isnull().sum().sum() == 0


def test_handle_missing_values_categorical():
    df = pd.DataFrame({
        "sex": ["male", np.nan, "male"],
        "cp": ["asymptomatic", "typical", np.nan]
    })

    result = handle_missing_values(df)

    assert result.isnull().sum().sum() == 0


def test_encode_categorical_variables():
    df = pd.DataFrame({
        "sex": ["female", "male"],
        "cp": ["typical", "asymptomatic"]
    })

    result = encode_categorical_variables(df)

    assert "sex_male" in result.columns


def test_clean_text_columns():
    df = pd.DataFrame({
        "sex": [" Male ", " FEMALE "]
    })

    result = clean_text_columns(df)

    assert result["sex"].tolist() == ["male", "female"]


def test_function_does_not_modify_original_dataframe():
    df = pd.DataFrame({
        "sex": [" Male ", " FEMALE "]
    })

    original_df = df.copy(deep=True)

    clean_text_columns(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_invalid_input_raises_type_error():
    with pytest.raises(TypeError):
        handle_missing_values("not a dataframe")


def test_empty_dataframe_raises_value_error():
    with pytest.raises(ValueError):
        handle_missing_values(pd.DataFrame())