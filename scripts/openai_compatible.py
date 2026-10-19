"""Minimal OpenAI-compatible Chat Completions client using Python stdlib."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Iterator


class InferenceTimeout(TimeoutError):
    pass


class InferenceHTTPError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class OpenAICompatibleClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 45, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.api_key = api_key

    def chat_completion(self, payload: dict[str, Any], *, request_id: str | None = None) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if request_id:
            headers["X-Request-ID"] = request_id
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise InferenceHTTPError(f"upstream returned HTTP {exc.code}", status_code=exc.code) from None
        except (socket.timeout, TimeoutError) as exc:
            raise InferenceTimeout("upstream request timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise InferenceTimeout("upstream request timed out") from exc
            raise InferenceHTTPError("upstream connection failed") from None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InferenceHTTPError(f"upstream returned invalid JSON: {type(exc).__name__}") from exc

        try:
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise InferenceHTTPError("upstream response is missing choices[0].message.content") from None
        if not isinstance(content, str):
            raise InferenceHTTPError("upstream message content must be text")
        return content

    def chat_completion_stream(
        self, payload: dict[str, Any], *, request_id: str | None = None
    ) -> Iterator[str]:
        """Open an OpenAI-compatible SSE stream and yield actual content deltas.

        The HTTP request is opened before this method returns so connection and
        HTTP errors can still be mapped to an ordinary Flask status response.
        The returned iterator owns and closes the upstream response.
        """
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        body = json.dumps(stream_payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if request_id:
            headers["X-Request-ID"] = request_id
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            upstream = urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as exc:
            raise InferenceHTTPError(f"upstream returned HTTP {exc.code}", status_code=exc.code) from None
        except (socket.timeout, TimeoutError) as exc:
            raise InferenceTimeout("upstream request timed out") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise InferenceTimeout("upstream request timed out") from exc
            raise InferenceHTTPError("upstream connection failed") from None

        def deltas() -> Iterator[str]:
            try:
                for raw_line in upstream:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                        choices = event.get("choices", [])
                        delta = choices[0].get("delta", {}) if choices else {}
                        content = delta.get("content")
                    except (json.JSONDecodeError, AttributeError, IndexError, TypeError):
                        raise InferenceHTTPError("upstream returned an invalid stream event") from None
                    if content is None:
                        continue
                    if not isinstance(content, str):
                        raise InferenceHTTPError("upstream stream content must be text")
                    if content:
                        yield content
            except (socket.timeout, TimeoutError) as exc:
                raise InferenceTimeout("upstream stream timed out") from exc
            except UnicodeDecodeError as exc:
                raise InferenceHTTPError("upstream stream was not valid UTF-8") from exc
            finally:
                upstream.close()

        return deltas()
