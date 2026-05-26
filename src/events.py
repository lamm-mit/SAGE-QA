from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class SAGEEvent:
    type: str
    agent: Optional[str] = None
    content: Optional[str] = None
    tool: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    graph: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class EventEmitter:
    """Thread-safe event emitter used by /qa_stream.

    This emits only visible runtime events: speaker transitions, messages that
    AutoGen has already placed in the groupchat history, tool calls, tool
    results, graph updates, final answers, and errors. It does not expose any
    hidden chain-of-thought.
    """

    def __init__(self):
        self.queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._seen_message_keys: set[str] = set()
        self._lock = threading.Lock()

    def emit(self, event: SAGEEvent | Dict[str, Any]) -> None:
        if isinstance(event, SAGEEvent):
            payload = asdict(event)
        else:
            payload = dict(event)
        payload = {k: v for k, v in payload.items() if v is not None}
        payload.setdefault("ts", time.time())
        self.queue.put(payload)

    def emit_once_for_message(self, index: int, msg: Dict[str, Any]) -> None:
        # De-duplicate because AutoGen speaker selection can inspect the same
        # last message more than once.
        name = str(msg.get("name") or msg.get("role") or "unknown")
        role = str(msg.get("role") or "")
        content = msg.get("content", "")
        key = f"{index}:{name}:{role}:{hash(str(content)[:1000])}:{bool(msg.get('tool_calls'))}:{bool(msg.get('function_call'))}"
        with self._lock:
            if key in self._seen_message_keys:
                return
            self._seen_message_keys.add(key)
        self.emit(message_to_event(index, msg))

    def done(self) -> None:
        self.emit({"type": "done"})

    def error(self, exc: Exception) -> None:
        self.emit({"type": "error", "content": f"{type(exc).__name__}: {exc}"})


_current_emitter: Optional[EventEmitter] = None


def set_current_emitter(emitter: Optional[EventEmitter]) -> None:
    global _current_emitter
    _current_emitter = emitter


def get_current_emitter() -> Optional[EventEmitter]:
    return _current_emitter


def emit(event: SAGEEvent | Dict[str, Any]) -> None:
    emitter = get_current_emitter()
    if emitter is not None:
        emitter.emit(event)


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except Exception:
        return str(value)


def normalize_tool_calls(msg: Dict[str, Any]) -> list[dict[str, Any]]:
    calls = msg.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for call in calls:
        if isinstance(call, dict):
            out.append(_safe_jsonable(call))
        else:
            # Support OpenAI/pydantic-like tool-call objects.
            f = getattr(call, "function", None)
            out.append({
                "id": getattr(call, "id", None),
                "type": getattr(call, "type", "function"),
                "function": {
                    "name": getattr(f, "name", None) if f is not None else None,
                    "arguments": getattr(f, "arguments", None) if f is not None else None,
                },
            })
    return out


def message_to_event(index: int, msg: Dict[str, Any]) -> Dict[str, Any]:
    name = msg.get("name") or msg.get("role") or "unknown"
    role = msg.get("role", "")
    content = msg.get("content", "")

    event: Dict[str, Any] = {
        "type": "agent_message",
        "index": index,
        "agent": str(name),
        "metadata": {"role": str(role)},
    }

    if content is not None:
        event["content"] = str(content)

    tool_calls = normalize_tool_calls(msg)
    if tool_calls:
        event["type"] = "tool_call_message"
        event["metadata"]["tool_calls"] = tool_calls

    if msg.get("function_call"):
        event["type"] = "tool_call_message"
        event["metadata"]["function_call"] = _safe_jsonable(msg.get("function_call"))

    if role == "tool":
        event["type"] = "tool_result_message"

    return event


def emit_visible_groupchat_message(groupchat: Any) -> None:
    emitter = get_current_emitter()
    if emitter is None:
        return
    messages = getattr(groupchat, "messages", []) or []
    if not messages:
        return
    idx = len(messages) - 1
    msg = messages[idx]
    if isinstance(msg, dict):
        emitter.emit_once_for_message(idx, msg)


def emit_next_speaker(next_speaker: Any) -> None:
    if next_speaker is None:
        emit({"type": "next_speaker", "agent": None, "content": "Conversation finished."})
        return
    name = getattr(next_speaker, "name", str(next_speaker))
    emit({"type": "next_speaker", "agent": str(name)})


def sse_format(event: Dict[str, Any]) -> str:
    event_type = event.get("type", "message")
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"
