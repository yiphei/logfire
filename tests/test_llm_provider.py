"""Tests for the llm_provider module."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any
from unittest import mock

import pytest
from inline_snapshot import snapshot

import logfire
from logfire._internal.integrations.llm_providers.llm_provider import (
    instrument_llm_provider,
    maybe_suppress_instrumentation,
    record_streaming,
)
from logfire._internal.integrations.llm_providers.types import EndpointConfig, StreamState
from logfire._internal.utils import suppress_instrumentation
from logfire.propagate import get_context
from logfire.testing import TestExporter


class MockStreamState(StreamState):
    """A mock stream state for testing."""

    def __init__(self):
        self.chunks: list[str] = []

    def record_chunk(self, chunk: Any) -> None:
        if isinstance(chunk, str):
            self.chunks.append(chunk)

    def get_response_data(self) -> Any:
        return {'combined_chunk_content': ''.join(self.chunks), 'chunk_count': len(self.chunks)}


@dataclass
class MockOptions:
    """Mock options object that simulates FinalRequestOptions."""

    url: str = '/test'
    json_data: dict[str, Any] = field(default_factory=lambda: {'model': 'test-model'})


class MockSyncStream:
    """A mock sync stream for testing streaming behavior."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    def __stream__(self) -> Iterator[str]:
        for chunk in self._chunks:
            yield chunk


class MockAsyncStream:
    """A mock async stream for testing streaming behavior."""

    def __init__(self, chunks: list[str]):
        self._chunks = chunks

    async def __stream__(self) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


class MockSyncClient:
    """A mock sync client for testing."""

    _is_instrumented_by_logfire = False

    def __init__(self, response: Any = None, chunks: list[str] | None = None):
        self._response = response or {'result': 'success'}
        self._chunks = chunks or []

    def request(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('stream') and self._chunks:
            # Use the stream_cls passed by the instrumentation wrapper
            stream_cls = kwargs.get('stream_cls', MockSyncStream)
            return stream_cls(self._chunks)
        return self._response


class MockAsyncClient:
    """A mock async client for testing."""

    _is_instrumented_by_logfire = False

    def __init__(self, response: Any = None, chunks: list[str] | None = None):
        self._response = response or {'result': 'success'}
        self._chunks = chunks or []

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('stream') and self._chunks:
            # Use the stream_cls passed by the instrumentation wrapper
            stream_cls = kwargs.get('stream_cls', MockAsyncStream)
            return stream_cls(self._chunks)
        return self._response


def mock_get_endpoint_config(options: MockOptions) -> EndpointConfig:
    """Mock endpoint config function for testing."""
    return EndpointConfig(
        message_template='Test with {request_data[model]!r}',
        span_data={'request_data': options.json_data},
        stream_state_cls=MockStreamState,
    )


def mock_on_response(response: Any, span: logfire.LogfireSpan) -> Any:
    """Mock on_response function for testing."""
    span.set_attribute('response_data', response)
    return response


def mock_is_sync_client(client_type: type) -> bool:
    """Mock is_async_client function - returns False for sync clients."""
    return issubclass(client_type, MockAsyncClient)


def test_instrument_single_client(exporter: TestExporter) -> None:
    """Test instrumenting a single sync client."""
    client = MockSyncClient()

    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    assert client._is_instrumented_by_logfire is True
    assert hasattr(client, '_original_request_method')

    # Make a request
    result = client.request(options=MockOptions())
    assert result == {'result': 'success'}

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1
    assert spans[0]['name'] == "Test with {request_data[model]!r}"
    assert spans[0]['attributes']['logfire.tags'] == ('LLM',)

    # Uninstrument
    with context_manager:
        pass

    assert client._is_instrumented_by_logfire is False
    assert not hasattr(client, '_original_request_method')


def test_instrument_already_instrumented_client(exporter: TestExporter) -> None:
    """Test that already instrumented clients are skipped."""
    client = MockSyncClient()
    client._is_instrumented_by_logfire = True

    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    # Should return nullcontext since already instrumented
    assert isinstance(context_manager, nullcontext)


def test_instrument_multiple_clients(exporter: TestExporter) -> None:
    """Test instrumenting multiple clients at once."""
    client1 = MockSyncClient()
    client2 = MockSyncClient()

    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=[client1, client2],
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    assert client1._is_instrumented_by_logfire is True
    assert client2._is_instrumented_by_logfire is True

    # Uninstrument both
    with context_manager:
        pass

    assert client1._is_instrumented_by_logfire is False
    assert client2._is_instrumented_by_logfire is False


async def test_instrument_async_client(exporter: TestExporter) -> None:
    """Test instrumenting an async client."""
    client = MockAsyncClient()

    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    assert client._is_instrumented_by_logfire is True

    # Make a request
    result = await client.request(options=MockOptions())
    assert result == {'result': 'success'}

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1
    assert spans[0]['name'] == "Test with {request_data[model]!r}"
    assert spans[0]['attributes']['async'] is True

    # Uninstrument
    with context_manager:
        pass

    assert client._is_instrumented_by_logfire is False


def test_instrumentation_suppressed(exporter: TestExporter) -> None:
    """Test that instrumentation is skipped when suppressed."""
    client = MockSyncClient()

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    # Make a request with suppressed instrumentation
    with suppress_instrumentation():
        result = client.request(options=MockOptions())

    assert result == {'result': 'success'}

    # No spans should be created
    assert exporter.exported_spans_as_dict() == []


def test_empty_message_template_skips_span(exporter: TestExporter) -> None:
    """Test that requests with empty message template skip span creation."""
    client = MockSyncClient()

    def empty_endpoint_config(options: MockOptions) -> EndpointConfig:
        return EndpointConfig(message_template='', span_data={})

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=empty_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    result = client.request(options=MockOptions())
    assert result == {'result': 'success'}

    # No spans should be created due to empty message template
    assert exporter.exported_spans_as_dict() == []


def test_maybe_suppress_instrumentation_true() -> None:
    """Test maybe_suppress_instrumentation when suppress is True."""
    from logfire._internal.utils import is_instrumentation_suppressed

    with maybe_suppress_instrumentation(True):
        assert is_instrumentation_suppressed() is True


def test_maybe_suppress_instrumentation_false() -> None:
    """Test maybe_suppress_instrumentation when suppress is False."""
    from logfire._internal.utils import is_instrumentation_suppressed

    with maybe_suppress_instrumentation(False):
        assert is_instrumentation_suppressed() is False


def test_record_streaming(exporter: TestExporter) -> None:
    """Test the record_streaming context manager."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE
    span_data = {'request_data': {'model': 'test-model'}}
    original_context = get_context()

    with record_streaming(logfire_instance, span_data, MockStreamState, original_context) as record_chunk:
        record_chunk('Hello ')
        record_chunk('World')
        record_chunk('')  # Empty chunk should be ignored

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1
    assert spans[0]['name'] == "streaming response from {request_data[model]!r} took {duration:.2f}s"
    assert 'duration' in spans[0]['attributes']
    assert spans[0]['attributes']['response_data'] == '{"combined_chunk_content":"Hello World","chunk_count":2}'


def test_record_streaming_preserves_context(exporter: TestExporter) -> None:
    """Test that record_streaming preserves the original context for sibling spans."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE

    with logfire_instance.span('parent span') as parent_span:
        # Capture context while inside the parent span
        original_context = get_context()
        span_data = {'request_data': {'model': 'test-model'}}

    # Now outside the parent span, simulate what happens in streaming:
    # The streaming response is processed after the original span ends
    with record_streaming(logfire_instance, span_data, MockStreamState, original_context) as record_chunk:
        record_chunk('chunk1')

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 2

    # Parent span
    parent = [s for s in spans if s['name'] == 'parent span'][0]
    # Streaming span
    streaming = [s for s in spans if 'streaming response' in s['name']][0]

    # The streaming span should be a child of the parent span (same trace, parent is set)
    assert streaming['context']['trace_id'] == parent['context']['trace_id']
    assert streaming['parent']['span_id'] == parent['context']['span_id']


def test_suppress_otel_flag(exporter: TestExporter) -> None:
    """Test that suppress_otel flag works correctly."""
    client = MockSyncClient()

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=True,  # This should suppress OTel instrumentation during the request
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    result = client.request(options=MockOptions())
    assert result == {'result': 'success'}

    # The span should still be created, but any nested OTel instrumentation should be suppressed
    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1


def test_client_with_underscore_request_method() -> None:
    """Test client that uses _request instead of request."""

    class ClientWithUnderscoreRequest:
        _is_instrumented_by_logfire = False

        def _request(self, *args: Any, **kwargs: Any) -> Any:
            return {'result': 'success'}

    client = ClientWithUnderscoreRequest()

    # Mock the try/except in instrument_llm_provider to use _request
    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=lambda _: False,
    )

    assert client._is_instrumented_by_logfire is True
    # For clients with _request, the attr_name should be _request
    # Verify by checking if _original_request_method exists
    assert hasattr(client, '_original_request_method')

    with context_manager:
        pass

    assert client._is_instrumented_by_logfire is False


def test_uninstrument_restores_original_method(exporter: TestExporter) -> None:
    """Test that uninstrumenting restores the original request method."""
    client = MockSyncClient()
    original_method = client.request

    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    # Method should be replaced
    assert client.request != original_method

    # Uninstrument
    with context_manager:
        pass

    # Method should be restored
    assert client.request == original_method


def test_span_data_includes_async_flag_sync(exporter: TestExporter) -> None:
    """Test that span_data includes async=False for sync clients."""
    client = MockSyncClient()

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    client.request(options=MockOptions())

    spans = exporter.exported_spans_as_dict()
    assert spans[0]['attributes']['async'] is False


async def test_span_data_includes_async_flag_async(exporter: TestExporter) -> None:
    """Test that span_data includes async=True for async clients."""
    client = MockAsyncClient()

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    await client.request(options=MockOptions())

    spans = exporter.exported_spans_as_dict()
    assert spans[0]['attributes']['async'] is True


def test_instrument_client_class(exporter: TestExporter) -> None:
    """Test instrumenting a client class (not an instance)."""

    class TestClientClass:
        _is_instrumented_by_logfire = False

        def request(self, *args: Any, **kwargs: Any) -> Any:
            return {'result': 'from class'}

    context_manager = instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=TestClientClass,  # Pass class, not instance
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=lambda cls: False,
    )

    assert TestClientClass._is_instrumented_by_logfire is True

    # Create instance and make request
    instance = TestClientClass()
    result = instance.request(options=MockOptions())
    assert result == {'result': 'from class'}

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1

    with context_manager:
        pass

    assert TestClientClass._is_instrumented_by_logfire is False


def test_custom_scope_suffix(exporter: TestExporter) -> None:
    """Test that custom scope suffix is applied correctly."""
    client = MockSyncClient()

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='CustomProvider',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    client.request(options=MockOptions())

    spans = exporter.exported_spans_as_dict(include_resources=True)
    assert len(spans) == 1
    # The scope suffix should be lowercased
    # The LLM tag should be present
    assert spans[0]['attributes']['logfire.tags'] == ('LLM',)


def test_sync_streaming_request(exporter: TestExporter) -> None:
    """Test sync client with streaming request."""
    chunks = ['Hello ', 'World', '!']
    client = MockSyncClient(chunks=chunks)

    def get_streaming_endpoint_config(options: MockOptions) -> EndpointConfig:
        return EndpointConfig(
            message_template='Test with {request_data[model]!r}',
            span_data={'request_data': options.json_data},
            stream_state_cls=MockStreamState,
        )

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=get_streaming_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    result = client.request(options=MockOptions(), stream=True, stream_cls=MockSyncStream)

    # Consume the stream
    collected = []
    for chunk in result.__stream__():
        collected.append(chunk)

    assert collected == chunks

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 2

    # First span is the request span
    request_span = spans[0]
    assert request_span['name'] == "Test with {request_data[model]!r}"

    # Second span is the streaming response span
    streaming_span = spans[1]
    assert 'streaming response' in streaming_span['name']
    assert 'duration' in streaming_span['attributes']


async def test_async_streaming_request(exporter: TestExporter) -> None:
    """Test async client with streaming request."""
    chunks = ['Hello ', 'World', '!']
    client = MockAsyncClient(chunks=chunks)

    def get_streaming_endpoint_config(options: MockOptions) -> EndpointConfig:
        return EndpointConfig(
            message_template='Test with {request_data[model]!r}',
            span_data={'request_data': options.json_data},
            stream_state_cls=MockStreamState,
        )

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=get_streaming_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    result = await client.request(options=MockOptions(), stream=True, stream_cls=MockAsyncStream)

    # Consume the stream
    collected = []
    async for chunk in result.__stream__():
        collected.append(chunk)

    assert collected == chunks

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 2

    # First span is the request span
    request_span = spans[0]
    assert request_span['name'] == "Test with {request_data[model]!r}"
    assert request_span['attributes']['async'] is True

    # Second span is the streaming response span
    streaming_span = spans[1]
    assert 'streaming response' in streaming_span['name']


def test_streaming_context_preserves_original_parent(exporter: TestExporter) -> None:
    """Test that the streaming log preserves the original context (parent span).

    This specifically tests the context propagation fix where get_context() is called
    inside _instrumentation_setup and passed to record_streaming. The context is captured
    BEFORE the request span is opened, so the streaming span and request span are siblings
    (both children of the same parent).
    """
    chunks = ['chunk1', 'chunk2']
    client = MockSyncClient(chunks=chunks)

    def get_streaming_endpoint_config(options: MockOptions) -> EndpointConfig:
        return EndpointConfig(
            message_template='Test with {request_data[model]!r}',
            span_data={'request_data': options.json_data},
            stream_state_cls=MockStreamState,
        )

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=get_streaming_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    # Create a parent span to test context propagation
    with logfire.span('parent operation'):
        result = client.request(options=MockOptions(), stream=True, stream_cls=MockSyncStream)

        # Consume the stream
        for _ in result.__stream__():
            pass

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 3

    # Find each span
    parent_span = [s for s in spans if s['name'] == 'parent operation'][0]
    request_span = [s for s in spans if s['name'] == "Test with {request_data[model]!r}"][0]
    streaming_span = [s for s in spans if 'streaming response' in s['name']][0]

    # All spans should be in the same trace
    assert request_span['context']['trace_id'] == parent_span['context']['trace_id']
    assert streaming_span['context']['trace_id'] == parent_span['context']['trace_id']

    # Request span should be a child of parent span
    assert request_span['parent']['span_id'] == parent_span['context']['span_id']

    # Streaming span should also be a child of the parent span (context captured before request span opened)
    # This means the streaming span and request span are siblings
    assert streaming_span['parent']['span_id'] == parent_span['context']['span_id']


def test_streaming_without_stream_state_cls(exporter: TestExporter) -> None:
    """Test streaming request when no stream_state_cls is provided."""
    chunks = ['Hello ', 'World']
    client = MockSyncClient(chunks=chunks)

    def get_no_stream_state_config(options: MockOptions) -> EndpointConfig:
        # No stream_state_cls provided
        return EndpointConfig(
            message_template='Test with {request_data[model]!r}',
            span_data={'request_data': options.json_data},
            stream_state_cls=None,
        )

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=get_no_stream_state_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_sync_client,
    )

    result = client.request(options=MockOptions(), stream=True, stream_cls=MockSyncStream)

    # The stream should be returned as-is (not wrapped)
    # Consume it to verify it works
    collected = list(result.__stream__())
    assert collected == chunks

    spans = exporter.exported_spans_as_dict()
    # Only the request span, no streaming span (because no stream_state_cls)
    assert len(spans) == 1
    assert spans[0]['name'] == "Test with {request_data[model]!r}"


def test_record_streaming_with_empty_chunks(exporter: TestExporter) -> None:
    """Test record_streaming handles empty chunks correctly."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE
    span_data = {'request_data': {'model': 'test-model'}}
    original_context = get_context()

    with record_streaming(logfire_instance, span_data, MockStreamState, original_context) as record_chunk:
        record_chunk('')  # Empty
        record_chunk(None)  # None (will be falsy)
        record_chunk('actual')
        record_chunk('')  # Empty again

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1
    # Only 'actual' should be recorded since empty/None chunks are ignored
    assert spans[0]['attributes']['response_data'] == '{"combined_chunk_content":"actual","chunk_count":1}'


class CustomStreamState(StreamState):
    """Custom stream state with custom get_attributes."""

    def __init__(self):
        self.chunks: list[str] = []

    def record_chunk(self, chunk: Any) -> None:
        if chunk:
            self.chunks.append(str(chunk))

    def get_response_data(self) -> Any:
        return {'text': ''.join(self.chunks)}

    def get_attributes(self, span_data: dict[str, Any]) -> dict[str, Any]:
        # Custom attributes that add extra data
        return {
            **span_data,
            'response_data': self.get_response_data(),
            'custom_metric': len(self.chunks),
        }


def test_record_streaming_custom_attributes(exporter: TestExporter) -> None:
    """Test that custom StreamState.get_attributes is used."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE
    span_data = {'request_data': {'model': 'test-model'}}
    original_context = get_context()

    with record_streaming(logfire_instance, span_data, CustomStreamState, original_context) as record_chunk:
        record_chunk('a')
        record_chunk('b')
        record_chunk('c')

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 1
    # Custom attributes should be present
    assert spans[0]['attributes']['custom_metric'] == 3
    assert spans[0]['attributes']['response_data'] == '{"text":"abc"}'
