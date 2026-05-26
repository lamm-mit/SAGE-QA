# ============================================================
# Autogen × vLLM (OpenAI-compatible) minimal safety patch
# Copied from the original notebook with only idempotency added.
# - Only patches OpenAIWrapper.create
# - Keeps Autogen logic unchanged
# - Focus:
#     1) sanitize outbound messages to backend
#     2) flatten weird list-of-parts content on return
#     3) optionally recover JSON-style tool calls from content
# ============================================================

from __future__ import annotations

import ast
import json
from autogen.oai.client import OpenAIWrapper

DEBUG_VLLM_PRE = False
DEBUG_VLLM_POST = False

_PATCHED = False
_ORIG_CREATE = None


def _as_text(content) -> str:
    """Convert message content into plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif isinstance(p.get("content"), str):
                    parts.append(p["content"])
                else:
                    parts.append(str(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)

    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return str(content)

    return str(content)


def _strip_one_wrapper_layer(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip()
    if t.startswith('"{{') and t.endswith('}}"') and len(t) >= 6:
        return t[2:-2].strip()
    if t.startswith("{{") and t.endswith("}}") and len(t) >= 4:
        return t[1:-1].strip()
    return t


def _flatten_list_of_parts_string(s: str) -> str:
    if not isinstance(s, str):
        return _as_text(s)
    original = s
    t = _strip_one_wrapper_layer(s).strip()
    if "type" not in t or "text" not in t:
        return original
    try:
        parsed = json.loads(t)
        if isinstance(parsed, str):
            try:
                parsed2 = json.loads(parsed)
                return _as_text(parsed2)
            except Exception:
                try:
                    parsed2 = ast.literal_eval(parsed)
                    return _as_text(parsed2)
                except Exception:
                    return parsed
        return _as_text(parsed)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(t)
        return _as_text(parsed)
    except Exception:
        return original


def _sanitize_messages(messages):
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        m2 = dict(m)
        role = str(m2.get("role", "assistant")).lower()
        if role not in ("system", "user", "assistant", "tool"):
            role = "assistant"
        m2["role"] = role
        m2["content"] = _as_text(m2.get("content", ""))
        if role == "tool" and m2["content"] is None:
            m2["content"] = ""
        tc = m2.get("tool_calls")
        if isinstance(tc, list) and len(tc) == 0:
            m2.pop("tool_calls", None)
        cleaned.append(m2)
    return cleaned


def _maybe_parse_json_or_literal_object(s: str):
    if not isinstance(s, str):
        return None
    t = _strip_one_wrapper_layer(s).strip()
    if not (t.startswith("{") and t.endswith("}")):
        return None
    try:
        data = json.loads(t)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    try:
        data = ast.literal_eval(t)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_tool_call_dict(data: dict):
    if not isinstance(data, dict):
        return None
    tc_raw = data.get("tool_calls")
    if isinstance(tc_raw, list) and len(tc_raw) > 0:
        return tc_raw
    name = None
    args_obj = {}
    if isinstance(data.get("name"), str) and data["name"].strip():
        name = data["name"].strip()
        for key in ("parameters", "arguments", "args", "input"):
            if key in data:
                args_obj = data[key]
                break
    if name is None and isinstance(data.get("function"), dict):
        fn = data["function"]
        if isinstance(fn.get("name"), str) and fn["name"].strip():
            name = fn["name"].strip()
            if "arguments" in fn:
                args_obj = fn["arguments"]
    if not name:
        return None
    if isinstance(args_obj, str):
        arguments_str = args_obj
    else:
        try:
            arguments_str = json.dumps(args_obj)
        except Exception:
            arguments_str = "{}"
    return [{
        "id": "call_from_text",
        "type": "function",
        "function": {"name": name, "arguments": arguments_str},
    }]


def _post_process_response(resp):
    try:
        msg = resp.choices[0].message
    except Exception:
        return resp

    existing_tc = getattr(msg, "tool_calls", None)
    if isinstance(existing_tc, list) and len(existing_tc) > 0:
        return resp

    raw_content = getattr(msg, "content", None)
    if isinstance(raw_content, list):
        flat = _as_text(raw_content)
        msg.content = flat
        text = flat
    elif isinstance(raw_content, str):
        flat = _flatten_list_of_parts_string(raw_content)
        msg.content = flat
        text = flat
    else:
        text = "" if raw_content is None else str(raw_content)
        msg.content = text

    text = (text or "").strip()
    if not text:
        return resp

    data = _maybe_parse_json_or_literal_object(text)
    if not data:
        return resp

    tool_calls = _extract_tool_call_dict(data)
    if not tool_calls:
        return resp

    try:
        msg.tool_calls = tool_calls
        msg.content = None
    except Exception:
        try:
            setattr(msg, "tool_calls", tool_calls)
            setattr(msg, "content", None)
        except Exception:
            return resp
    return resp


def apply_vllm_patch() -> None:
    global _PATCHED, _ORIG_CREATE
    if _PATCHED:
        return
    _ORIG_CREATE = OpenAIWrapper.create

    def _safe_create(self, **config):
        kw = dict(config)
        if DEBUG_VLLM_PRE:
            print("========= [PRE DEBUG] RAW CONFIG =========")
            try:
                print(repr(kw))
            except Exception as e:
                print(f"[repr failed: {e}]")
            print("==========================================")

        msgs = kw.get("messages")
        if isinstance(msgs, list):
            kw["messages"] = _sanitize_messages(msgs)

        kw.pop("parallel_tool_calls", None)
        kw.pop("functions", None)
        kw.pop("function_call", None)

        if DEBUG_VLLM_PRE:
            print("====== [PRE DEBUG] SANITIZED CONFIG ======")
            try:
                printable = {k: v for k, v in kw.items() if k not in ("agent", "cache")}
                print(json.dumps(printable, indent=2, ensure_ascii=False, default=str))
            except Exception as e:
                print(f"[json.dumps failed: {e}]")
            print("==========================================")

        resp = _ORIG_CREATE(self, **kw)

        if DEBUG_VLLM_POST:
            print("======= [POST DEBUG] RAW RESPONSE ========")
            try:
                print(resp)
            except Exception as e:
                print(f"[print(resp) failed: {e}]")
            print("==========================================")

        resp = _post_process_response(resp)

        if DEBUG_VLLM_POST:
            try:
                msg = resp.choices[0].message
                print("==== [POST DEBUG] MESSAGE TO AUTOGEN =====")
                print("role      :", getattr(msg, "role", None))
                print("content   :", repr(getattr(msg, "content", None)))
                print("tool_calls:", getattr(msg, "tool_calls", None))
                print("==========================================")
            except Exception as e:
                print(f"[POST inspect failed: {e}")
        return resp

    OpenAIWrapper.create = _safe_create
    _PATCHED = True
