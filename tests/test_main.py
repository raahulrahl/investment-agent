"""Tests for the Investment Agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from investment_agent.main import handler


@pytest.mark.asyncio
async def test_handler_returns_response():
    """Test that handler accepts messages and returns a response."""
    messages = [{"role": "user", "content": "Analyze AAPL stock"}]

    # Mock the run_agent function to return a mock response
    mock_response = MagicMock()
    mock_response.run_id = "test-run-id"
    mock_response.status = "COMPLETED"

    # Mock _initialized to skip initialization and run_agent to return our mock
    with (
        patch("investment_agent.main._initialized", True),
        patch("investment_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    # Verify we get a result back
    assert result is not None
    assert result.run_id == "test-run-id"
    assert result.status == "COMPLETED"


@pytest.mark.asyncio
async def test_handler_with_multiple_messages():
    """Test that handler processes multiple messages correctly."""
    messages = [
        {"role": "system", "content": "You are an investment analyst."},
        {"role": "user", "content": "Compare AAPL vs MSFT stocks"},
    ]

    mock_response = MagicMock()
    mock_response.run_id = "test-run-id-2"

    with (
        patch("investment_agent.main._initialized", True),
        patch("investment_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response) as mock_run,
    ):
        result = await handler(messages)

    # Verify run_agent was called
    mock_run.assert_called_once_with(messages)
    assert result is not None
    assert result.run_id == "test-run-id-2"


@pytest.mark.asyncio
async def test_handler_initialization():
    """Test that handler initializes on first call."""
    messages = [{"role": "user", "content": "What's the current price of TSLA?"}]

    mock_response = MagicMock()

    # Start with _initialized as False to test initialization path
    with (
        patch("investment_agent.main._initialized", False),
        patch("investment_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init,
        patch("investment_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response) as mock_run,
        patch("investment_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        result = await handler(messages)

        # Verify initialization was called
        mock_init.assert_called_once()
        # Verify run_agent was called
        mock_run.assert_called_once_with(messages)
        # Verify we got a result
        assert result is not None


@pytest.mark.asyncio
async def test_handler_race_condition_prevention():
    """Test that handler prevents race conditions with initialization lock."""
    messages = [{"role": "user", "content": "Analyze GOOGL fundamentals"}]

    mock_response = MagicMock()

    # Test with multiple concurrent calls
    with (
        patch("investment_agent.main._initialized", False),
        patch("investment_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init,
        patch("investment_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
        patch("investment_agent.main._init_lock", new_callable=MagicMock()) as mock_lock,
    ):
        # Configure the lock to work as an async context manager
        mock_lock_instance = MagicMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_lock.return_value = mock_lock_instance

        # Call handler twice to ensure lock is used
        await handler(messages)
        await handler(messages)

        # Verify initialize_agent was called only once (due to lock)
        mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_handler_with_investment_query():
    """Test that handler can process an investment analysis query."""
    messages = [
        {
            "role": "user",
            "content": "Provide comprehensive analysis of Apple (AAPL) stock for long-term investment",
        }
    ]

    mock_response = MagicMock()
    mock_response.run_id = "investment-run-id"
    mock_response.content = "Investment analysis report generated successfully."

    with (
        patch("investment_agent.main._initialized", True),
        patch("investment_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    assert result is not None
    assert result.run_id == "investment-run-id"
    assert result.content == "Investment analysis report generated successfully."


@pytest.mark.asyncio
async def test_handler_with_stock_comparison():
    """Test that handler can process stock comparison queries."""
    messages = [
        {
            "role": "user",
            "content": "Compare Microsoft (MSFT) and Google (GOOGL) stocks over the past year",
        }
    ]

    mock_response = MagicMock()
    mock_response.run_id = "comparison-run-id"
    mock_response.content = "Stock comparison analysis completed."

    with (
        patch("investment_agent.main._initialized", True),
        patch("investment_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    assert result is not None
    assert result.run_id == "comparison-run-id"
    assert result.content == "Stock comparison analysis completed."
