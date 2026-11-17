"""Request guards applied before FastAPI parses multipart form data."""

import json
from tempfile import SpooledTemporaryFile

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestSizeLimitMiddleware:
    """Bound total prediction request bodies without trusting Content-Length.

    The request is buffered in a spooled temporary file before it reaches
    FastAPI's multipart parser. This caps parser input even for chunked or
    headerless requests, while keeping only a small amount in memory.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, path: str = "/api/predict") -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path = path

    async def _reject(self, send: Send) -> None:
        payload = json.dumps(
            {"detail": "Request body exceeds the total upload size limit."}
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST" or scope.get("path") != self.path:
            await self.app(scope, receive, send)
            return

        # Reject an obviously oversized request early, but always count actual
        # bytes below because Content-Length can be absent or untrustworthy.
        content_length = next(
            (value for name, value in scope.get("headers", []) if name.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass

        total = 0
        with SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as body_file:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue

                chunk = message.get("body", b"")
                total += len(chunk)
                if total > self.max_bytes:
                    await self._reject(send)
                    return

                body_file.write(chunk)
                if not message.get("more_body", False):
                    break

            body_file.seek(0)
            remaining = total
            empty_body_sent = False

            async def replay_receive() -> Message:
                nonlocal remaining, empty_body_sent
                if remaining <= 0:
                    if not empty_body_sent:
                        empty_body_sent = True
                        return {"type": "http.request", "body": b"", "more_body": False}
                    return {"type": "http.disconnect"}
                chunk = body_file.read(min(64 * 1024, remaining))
                remaining -= len(chunk)
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": remaining > 0,
                }

            await self.app(scope, replay_receive, send)
