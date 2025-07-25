import json

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from utilsforecast.data import generate_series

from timecopilot.agent import ForecastAgentOutput, TimeCopilot


def build_stub_llm(output: dict) -> FunctionModel:  # noqa: D401
    def _response_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: D401
        payload = json.dumps(output)
        return ModelResponse(
            parts=[ToolCallPart(tool_name="final_result", args=payload)]
        )

    return FunctionModel(_response_fn)


@pytest.mark.parametrize("query", [None, "dummy"])
def test_forecast_returns_expected_output(query):
    df = generate_series(n_series=1, freq="D", min_length=30)
    expected_output = {
        "tsfeatures_analysis": "ok",
        "selected_model": "ZeroModel",
        "model_details": "details",
        "model_comparison": "cmp",
        "is_better_than_seasonal_naive": True,
        "reason_for_selection": "reason",
        "forecast_analysis": "analysis",
        "user_query_response": query,
    }
    tc = TimeCopilot(llm=build_stub_llm(expected_output))
    tc.fcst_df = None
    tc.eval_df = None
    tc.features_df = None
    result = tc.forecast(df=df, h=2, freq="D", seasonality=7, query=query)

    assert result.output == ForecastAgentOutput(**expected_output)


def test_constructor_rejects_model_kwarg():
    with pytest.raises(ValueError):
        TimeCopilot(llm="test", model="something")


def test_defensive_dataframe_assignment():
    """Test that forecast methods handle missing dataframes gracefully."""
    df = generate_series(n_series=1, freq="D", min_length=30)
    expected_output = {
        "tsfeatures_analysis": "ok",
        "selected_model": "ZeroModel",
        "model_details": "details",
        "model_comparison": "cmp",
        "is_better_than_seasonal_naive": True,
        "reason_for_selection": "reason",
        "forecast_analysis": "analysis",
        "user_query_response": None,
    }
    tc = TimeCopilot(llm=build_stub_llm(expected_output))
    
    # Don't set the dataframe attributes - they should be None by default
    result = tc.forecast(df=df, h=2, freq="D", seasonality=7)
    
    # The result should not have the dataframe attributes when they're None
    assert not hasattr(result, 'fcst_df') or result.fcst_df is None
    assert not hasattr(result, 'eval_df') or result.eval_df is None  
    assert not hasattr(result, 'features_df') or result.features_df is None
    assert result.output == ForecastAgentOutput(**expected_output)


def test_initialization_with_none_attributes():
    """Test that TimeCopilot initializes attributes to None."""
    tc = TimeCopilot(llm="test")
    
    # Check that all attributes are initialized to None
    assert tc.dataset is None
    assert tc.fcst_df is None
    assert tc.eval_df is None
    assert tc.features_df is None
    assert tc.models is None
    
    # Check that is_queryable returns False when attributes are None
    assert not tc.is_queryable()
