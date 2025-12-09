"""Tests for the llm_provider module - specifically context preservation during streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import logfire
from logfire._internal.integrations.llm_providers.llm_provider import (
    instrument_llm_provider,
    record_streaming,
)
from logfire._internal.integrations.llm_providers.types import EndpointConfig, StreamState
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

    def __init__(self, chunks: list[str] | None = None):
        self._chunks = chunks or []

    def request(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('stream') and self._chunks:
            stream_cls = kwargs.get('stream_cls', MockSyncStream)
            return stream_cls(self._chunks)
        return {'result': 'success'}


class MockAsyncClient:
    """A mock async client for testing."""

    _is_instrumented_by_logfire = False

    def __init__(self, chunks: list[str] | None = None):
        self._chunks = chunks or []

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('stream') and self._chunks:
            stream_cls = kwargs.get('stream_cls', MockAsyncStream)
            return stream_cls(self._chunks)
        return {'result': 'success'}


def mock_get_endpoint_config(options: MockOptions) -> EndpointConfig:
    return EndpointConfig(
        message_template='Test with {request_data[model]!r}',
        span_data={'request_data': options.json_data},
        stream_state_cls=MockStreamState,
    )


def mock_on_response(response: Any, span: logfire.LogfireSpan) -> Any:
    return response


def mock_is_async_client(client_type: type) -> bool:
    return issubclass(client_type, MockAsyncClient)


def test_record_streaming_preserves_context(exporter: TestExporter) -> None:
    """Test that record_streaming uses attach_context to preserve the original context."""
    logfire_instance = logfire.DEFAULT_LOGFIRE_INSTANCE

    with logfire_instance.span('parent span'):
        # Capture context while inside the parent span
        original_context = get_context()
        span_data = {'request_data': {'model': 'test-model'}}

    # Now outside the parent span, the streaming log should still be linked to the parent
    with record_streaming(logfire_instance, span_data, MockStreamState, original_context) as record_chunk:
        record_chunk('chunk1')

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 2

    parent = [s for s in spans if s['name'] == 'parent span'][0]
    streaming = [s for s in spans if 'streaming response' in s['name']][0]

    # The streaming span should be a child of the parent span
    assert streaming['context']['trace_id'] == parent['context']['trace_id']
    assert streaming['parent']['span_id'] == parent['context']['span_id']


def test_sync_streaming_preserves_original_context(exporter: TestExporter) -> None:
    """Test that sync streaming requests preserve the original context.

    The context is captured in _instrumentation_setup (before the request span opens),
    so the streaming log and request span are siblings under the same parent.
    """
    client = MockSyncClient(chunks=['chunk1', 'chunk2'])

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_async_client,
    )

    with logfire.span('parent operation'):
        result = client.request(options=MockOptions(), stream=True, stream_cls=MockSyncStream)
        for _ in result.__stream__():
            pass

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 3

    parent_span = [s for s in spans if s['name'] == 'parent operation'][0]
    request_span = [s for s in spans if s['name'] == "Test with {request_data[model]!r}"][0]
    streaming_span = [s for s in spans if 'streaming response' in s['name']][0]

    # All spans in the same trace
    assert request_span['context']['trace_id'] == parent_span['context']['trace_id']
    assert streaming_span['context']['trace_id'] == parent_span['context']['trace_id']

    # Request span is child of parent
    assert request_span['parent']['span_id'] == parent_span['context']['span_id']

    # Streaming span is also child of parent (siblings with request span)
    assert streaming_span['parent']['span_id'] == parent_span['context']['span_id']


async def test_async_streaming_preserves_original_context(exporter: TestExporter) -> None:
    """Test that async streaming requests preserve the original context."""
    client = MockAsyncClient(chunks=['chunk1', 'chunk2'])

    instrument_llm_provider(
        logfire=logfire.DEFAULT_LOGFIRE_INSTANCE,
        client=client,
        suppress_otel=False,
        scope_suffix='test',
        get_endpoint_config_fn=mock_get_endpoint_config,
        on_response_fn=mock_on_response,
        is_async_client_fn=mock_is_async_client,
    )

    with logfire.span('parent operation'):
        result = await client.request(options=MockOptions(), stream=True, stream_cls=MockAsyncStream)
        async for _ in result.__stream__():
            pass

    spans = exporter.exported_spans_as_dict()
    assert len(spans) == 3

    parent_span = [s for s in spans if s['name'] == 'parent operation'][0]
    request_span = [s for s in spans if s['name'] == "Test with {request_data[model]!r}"][0]
    streaming_span = [s for s in spans if 'streaming response' in s['name']][0]

    # All spans in the same trace
    assert request_span['context']['trace_id'] == parent_span['context']['trace_id']
    assert streaming_span['context']['trace_id'] == parent_span['context']['trace_id']

    # Request span is child of parent
    assert request_span['parent']['span_id'] == parent_span['context']['span_id']

    # Streaming span is also child of parent (siblings with request span)
    assert streaming_span['parent']['span_id'] == parent_span['context']['span_id']
