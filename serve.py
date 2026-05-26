from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


# ---------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import SAGEQAApp  # noqa: E402
from config import SAGEConfig  # noqa: E402


# ---------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------

QA_APP: Optional[SAGEQAApp] = None
APP_ARGS: Optional[argparse.Namespace] = None
STREAM_JOBS: Dict[str, Dict[str, Any]] = {}

# Use spawn, not fork. Fork breaks CUDA once torch/CUDA has been initialized.
MP_CTX = mp.get_context("spawn")


# ---------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------

class QARequest(BaseModel):
    question: str


class QAStreamRequest(BaseModel):
    question: str
    job_id: Optional[str] = None


# ---------------------------------------------------------------------
# Config / app construction
# ---------------------------------------------------------------------

def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    data = vars(args).copy()

    data["data_dir"] = Path(data["data_dir"]).expanduser()
    data["data_dir_out"] = Path(data["data_dir_out"]).expanduser()
    data["project_root"] = PROJECT_ROOT
    data["agent_settings_dir"] = PROJECT_ROOT / "agent_settings"

    return argparse.Namespace(**data)


def build_config(args: argparse.Namespace) -> SAGEConfig:
    args = normalize_args(args)

    if hasattr(SAGEConfig, "from_args"):
        return SAGEConfig.from_args(args)

    if hasattr(SAGEConfig, "from_env_and_args"):
        return SAGEConfig.from_env_and_args(args)

    kwargs = {
        "base_url": args.base_url,
        "model": args.model,
        "data_dir": args.data_dir,
        "data_dir_out": args.data_dir_out,
        "project_root": args.project_root,
        "agent_settings_dir": args.agent_settings_dir,
    }

    sig = inspect.signature(SAGEConfig)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    alt_map = {
        "llm_base_url": args.base_url,
        "qa_base_url": args.base_url,
        "model_name": args.model,
        "qa_model": args.model,
        "data_path": args.data_dir,
        "output_dir": args.data_dir_out,
        "data_output_dir": args.data_dir_out,
        "settings_dir": args.agent_settings_dir,
    }

    for k, v in alt_map.items():
        if k in accepted and k not in filtered:
            filtered[k] = v

    return SAGEConfig(**filtered)


def build_qa_app(args: argparse.Namespace) -> SAGEQAApp:
    config = build_config(args)
    qa_app = SAGEQAApp(config)

    if hasattr(qa_app, "setup"):
        qa_app.setup()

    return qa_app


def get_parent_qa_app() -> SAGEQAApp:
    global QA_APP

    if QA_APP is None:
        if APP_ARGS is None:
            raise RuntimeError("APP_ARGS is not initialized.")
        QA_APP = build_qa_app(APP_ARGS)

    return QA_APP


def ask_blocking(question: str) -> str:
    qa_app = get_parent_qa_app()

    if hasattr(qa_app, "ask"):
        return str(qa_app.ask(question))

    if hasattr(qa_app, "run"):
        return str(qa_app.run(question))

    if hasattr(qa_app, "qa"):
        return str(qa_app.qa(question))

    raise RuntimeError("SAGEQAApp has no ask(), run(), or qa() method.")


def call_ask_stream(qa_app: SAGEQAApp, question: str, emitter: Any) -> Optional[str]:
    if not hasattr(qa_app, "ask_stream"):
        return None

    fn = getattr(qa_app, "ask_stream")
    sig = inspect.signature(fn)
    params = sig.parameters

    if "question" in params and "emitter" in params:
        return fn(question=question, emitter=emitter)

    if "question" in params and "event_emitter" in params:
        return fn(question=question, event_emitter=emitter)

    if "query" in params and "emitter" in params:
        return fn(query=question, emitter=emitter)

    if len(params) == 2:
        return fn(question, emitter)

    if len(params) == 1:
        return fn(question)

    return fn(question=question, emitter=emitter)


def make_worker_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "data_dir": str(Path(args.data_dir).expanduser()),
        "data_dir_out": str(Path(args.data_dir_out).expanduser()),
        "host": args.host,
        "port": args.port,
    }


# ---------------------------------------------------------------------
# SSE / event helpers
# ---------------------------------------------------------------------

def now_ts() -> float:
    return time.time()


def clean_event(event: Dict[str, Any]) -> Dict[str, Any]:
    event = dict(event)
    event = {k: v for k, v in event.items() if v is not None}
    event.setdefault("ts", now_ts())
    return event


def sse_format(event: Dict[str, Any]) -> str:
    event = clean_event(event)
    event_type = event.get("type", "message")
    data = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


class ProcessEmitter:
    """
    Event emitter used inside the spawned QA worker process.

    Supports the interface expected by existing app/core/factory/tools code:
    - emit(...)
    - emit_once_for_message(...)
    - emit_next_speaker(...)
    - emit_tool_call(...)
    - emit_tool_result(...)
    - emit_graph_update(...)
    - emit_status(...)
    - emit_final_answer(...)
    - done()
    - error(...)
    """

    def __init__(self, q):
        self.q = q
        self._seen_message_keys = set()

    def emit(self, event: Any) -> None:
        if dataclasses.is_dataclass(event):
            payload = dataclasses.asdict(event)
        elif isinstance(event, dict):
            payload = dict(event)
        elif hasattr(event, "__dict__"):
            payload = dict(event.__dict__)
        else:
            payload = {
                "type": "message",
                "content": str(event),
            }

        self.q.put(clean_event(payload))

    def emit_once_for_message(self, *args, **kwargs) -> None:
        index = (
            kwargs.get("index", None)
            if kwargs.get("index", None) is not None
            else kwargs.get("message_index", None)
        )

        message = (
            kwargs.get("message", None)
            or kwargs.get("msg", None)
            or kwargs.get("event", None)
        )

        start_index = kwargs.get("start_index", None)

        if message is None:
            if len(args) == 1:
                message = args[0]
            elif len(args) >= 2:
                if isinstance(args[0], int):
                    index = args[0]
                    message = args[1]
                else:
                    message = args[0]
                    if isinstance(args[1], int):
                        index = args[1]

        if message is None:
            return

        if isinstance(message, list):
            base = start_index if start_index is not None else (index or 0)
            for offset, msg in enumerate(message):
                self.emit_once_for_message(msg, index=base + offset)
            return

        if not isinstance(message, dict):
            key = f"{index}|unknown|{str(message)}"
            if key in self._seen_message_keys:
                return

            self._seen_message_keys.add(key)

            self.emit({
                "type": "agent_message",
                "index": index,
                "agent": "unknown",
                "content": str(message),
            })
            return

        agent = (
            message.get("name")
            or message.get("agent")
            or message.get("sender")
            or message.get("role")
            or "unknown"
        )

        role = message.get("role", "")
        content = message.get("content", "")

        tool_calls = message.get("tool_calls") or message.get("function_call")
        tool_responses = message.get("tool_responses")

        key = json.dumps(
            {
                "index": index,
                "agent": agent,
                "role": role,
                "content": content,
                "tool_calls": tool_calls,
                "tool_responses": tool_responses,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        if key in self._seen_message_keys:
            return

        self._seen_message_keys.add(key)

        metadata = {"role": role}

        if tool_calls:
            metadata["tool_calls"] = tool_calls

        if tool_responses:
            metadata["tool_responses"] = tool_responses

        if tool_calls:
            self.emit({
                "type": "tool_call_message",
                "index": index,
                "agent": agent,
                "metadata": metadata,
                "content": content,
            })
        else:
            self.emit({
                "type": "agent_message",
                "index": index,
                "agent": agent,
                "metadata": metadata,
                "content": content,
            })

    def emit_next_speaker(self, *args, **kwargs) -> None:
        speaker = kwargs.get("speaker", None) or kwargs.get("agent", None)

        if speaker is None:
            if len(args) == 1:
                speaker = args[0]
            elif len(args) >= 2:
                speaker = args[-1]

        if speaker is None:
            return

        name = getattr(speaker, "name", None) or str(speaker)

        self.emit({
            "type": "next_speaker",
            "agent": name,
        })

    def emit_tool_call(
        self,
        tool: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        agent: str = "engineer",
        **kwargs,
    ) -> None:
        tool = tool or kwargs.get("name") or kwargs.get("tool_name") or "tool"
        arguments = arguments or kwargs.get("args") or kwargs.get("arguments") or {}

        self.emit({
            "type": "tool_call",
            "agent": agent,
            "tool": tool,
            "arguments": arguments,
        })

    def emit_tool_result(
        self,
        tool: Optional[str] = None,
        result: Any = "",
        agent: str = "user_proxy",
        max_chars: int = 12000,
        **kwargs,
    ) -> None:
        tool = tool or kwargs.get("name") or kwargs.get("tool_name") or "tool"
        result = result or kwargs.get("content") or kwargs.get("output") or ""

        text = str(result)

        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
            truncated = True

        self.emit({
            "type": "tool_result",
            "agent": agent,
            "tool": tool,
            "result": text,
            "metadata": {"truncated": truncated},
        })

    def emit_graph_update(self, graph: Dict[str, Any]) -> None:
        # Kept for backend compatibility. The current UI intentionally ignores graph_update.
        self.emit({
            "type": "graph_update",
            "agent": "engineer",
            "graph": graph,
        })

    def emit_status(self, content: str, agent: str = "system", **metadata) -> None:
        self.emit({
            "type": "status",
            "agent": agent,
            "content": content,
            "metadata": metadata or None,
        })

    def emit_final_answer(self, content: str, agent: str = "summarizer", **metadata) -> None:
        self.emit({
            "type": "final_answer",
            "agent": agent,
            "content": content,
            "metadata": metadata or None,
        })

    def done(self) -> None:
        self.emit({"type": "done"})

    def error(self, exc: BaseException) -> None:
        self.emit({
            "type": "error",
            "agent": "system",
            "content": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def qa_stream_worker(job_id: str, question: str, q, worker_config: Dict[str, Any]) -> None:
    emitter = ProcessEmitter(q)

    try:
        emitter.emit({
            "type": "status",
            "agent": "system",
            "content": "Initializing SAGE-QA worker process...",
            "metadata": {"job_id": job_id},
        })

        child_args = argparse.Namespace(**worker_config)
        qa_app = build_qa_app(child_args)

        emitter.emit({
            "type": "run_start",
            "agent": "user",
            "content": question,
            "metadata": {"job_id": job_id},
        })

        answer = call_ask_stream(qa_app=qa_app, question=question, emitter=emitter)

        if answer is None:
            emitter.emit({
                "type": "status",
                "agent": "system",
                "content": "ask_stream() not found; falling back to blocking ask().",
                "metadata": {"job_id": job_id},
            })

            if hasattr(qa_app, "ask"):
                answer = str(qa_app.ask(question))
            elif hasattr(qa_app, "run"):
                answer = str(qa_app.run(question))
            elif hasattr(qa_app, "qa"):
                answer = str(qa_app.qa(question))
            else:
                raise RuntimeError("SAGEQAApp has no ask_stream(), ask(), run(), or qa() method.")

        if answer:
            emitter.emit({
                "type": "final_answer",
                "agent": "summarizer",
                "content": str(answer),
                "metadata": {"job_id": job_id},
            })

    except BaseException as exc:
        emitter.error(exc)

    finally:
        emitter.done()


def terminate_stream_job(job_id: str) -> Dict[str, Any]:
    record = STREAM_JOBS.get(job_id)

    if not record:
        return {
            "ok": False,
            "job_id": job_id,
            "message": "Job not found or already finished.",
        }

    proc = record["process"]
    q = record["queue"]

    try:
        q.put(clean_event({
            "type": "cancelled",
            "agent": "system",
            "content": f"Stop requested for job {job_id}.",
            "metadata": {"job_id": job_id},
        }))
        q.put(clean_event({
            "type": "done",
            "metadata": {"job_id": job_id},
        }))
    except Exception:
        pass

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=2)

    STREAM_JOBS.pop(job_id, None)

    return {
        "ok": True,
        "job_id": job_id,
        "message": "Job stopped.",
    }


# ---------------------------------------------------------------------
# Native HTML UI
# ---------------------------------------------------------------------

UI_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SAGE-QA Live Web UI</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: #f7f7f8;
      color: #111827;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    h1 {
      margin: 0 0 16px 0;
      font-size: 28px;
    }
    textarea {
      width: 100%;
      min-height: 105px;
      padding: 12px;
      font-size: 15px;
      border: 1px solid #d1d5db;
      border-radius: 10px;
      resize: vertical;
      box-sizing: border-box;
      background: white;
    }
    button {
      padding: 10px 18px;
      margin: 10px 8px 10px 0;
      border: 0;
      border-radius: 10px;
      cursor: pointer;
      font-size: 15px;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    #askBtn {
      background: #111827;
      color: white;
    }
    #stopBtn {
      background: #dc2626;
      color: white;
    }
    #clearBtn {
      background: #e5e7eb;
      color: #111827;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 16px;
    }
    .panel {
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 14px;
      padding: 16px;
      min-height: 280px;
      max-height: 520px;
      overflow: auto;
      white-space: pre-wrap;
      line-height: 1.45;
    }
    .panel h2 {
      margin-top: 0;
      font-size: 18px;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 8px;
    }
    #answer {
      min-height: 180px;
      max-height: 480px;
    }
    .full {
      grid-column: span 2;
    }
    .event-title {
      font-weight: 700;
      color: #2563eb;
      margin-top: 12px;
    }
    .small {
      color: #6b7280;
      font-size: 13px;
    }
    code {
      background: #f3f4f6;
      padding: 2px 4px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>SAGE-QA Live Web UI</h1>

    <textarea id="question" placeholder="Ask a technical question..."></textarea>

    <div>
      <button id="askBtn" onclick="ask()">Ask</button>
      <button id="stopBtn" onclick="stopJob()">Stop</button>
      <button id="clearBtn" onclick="clearAll()">Clear</button>
      <span id="status" class="small"></span>
    </div>

    <div class="panel full" id="answer">
      <h2>Final Answer</h2>
      <div id="answerContent">Final answer will appear here.</div>
    </div>

    <div class="grid">
      <div class="panel">
        <h2>Agent Trace</h2>
        <div id="agentTrace"></div>
      </div>

      <div class="panel">
        <h2>Tool Events</h2>
        <div id="toolEvents"></div>
      </div>

      <div class="panel full">
        <h2>Raw Events</h2>
        <div id="rawEvents"></div>
      </div>
    </div>
  </div>

<script>
let currentJobId = null;
let eventSource = null;

function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function setStatus(text) {
  document.getElementById("status").textContent = text || "";
}

function setAnswer(text) {
  document.getElementById("answerContent").textContent = text || "";
  const panel = document.getElementById("answer");
  panel.scrollTop = panel.scrollHeight;
}

function append(id, title, content) {
  const el = document.getElementById(id);
  const block = document.createElement("div");

  const titleEl = document.createElement("div");
  titleEl.className = "event-title";
  titleEl.textContent = title;

  const contentEl = document.createElement("div");
  contentEl.textContent = content || "";

  block.appendChild(titleEl);
  block.appendChild(contentEl);
  el.appendChild(block);

  const panel = el.closest(".panel");
  panel.scrollTop = panel.scrollHeight;
}

function appendRaw(data) {
  const el = document.getElementById("rawEvents");
  el.textContent += JSON.stringify(data, null, 2) + "\n\n";
  const panel = el.closest(".panel");
  panel.scrollTop = panel.scrollHeight;
}

function setRunning(isRunning) {
  document.getElementById("askBtn").disabled = isRunning;
  document.getElementById("stopBtn").disabled = !isRunning;
}

function clearAll() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  document.getElementById("question").value = "";
  document.getElementById("agentTrace").innerHTML = "";
  document.getElementById("toolEvents").innerHTML = "";
  document.getElementById("rawEvents").textContent = "";
  setAnswer("Final answer will appear here.");
  setStatus("");
  currentJobId = null;
  setRunning(false);
}

function handleEvent(eventType, data) {
  const etype = data.type || eventType || "message";

  if (etype === "heartbeat") {
    return;
  }

  appendRaw(data);

  if (etype === "job_start") {
    append("agentTrace", "Job Start", data.content || "");
  } else if (etype === "run_start") {
    append("agentTrace", "Run Start", data.content || "");
  } else if (etype === "status") {
    append("agentTrace", "Status", data.content || "");
  } else if (etype === "next_speaker") {
    append("agentTrace", "Next Speaker", data.agent || "");
  } else if (etype === "agent_message") {
    append("agentTrace", "Agent: " + (data.agent || "unknown"), data.content || "");
  } else if (etype === "tool_call_message") {
    append(
      "toolEvents",
      "Tool Call Message from " + (data.agent || "engineer"),
      JSON.stringify(data.metadata || {}, null, 2)
    );
  } else if (etype === "tool_call") {
    append(
      "toolEvents",
      "Tool Call: " + (data.tool || "tool") + " by " + (data.agent || ""),
      JSON.stringify(data.arguments || {}, null, 2)
    );
  } else if (etype === "tool_result") {
    append("toolEvents", "Tool Result: " + (data.tool || "tool"), data.result || data.content || "");
  } else if (etype === "final_answer" || etype === "final") {
    setAnswer(data.answer || data.content || data.result || "");
  } else if (etype === "cancelled") {
    setAnswer(data.content || "Stopped.");
    setStatus("Stopped.");
    currentJobId = null;
    setRunning(false);
  } else if (etype === "error") {
    setAnswer("Error:\n\n" + (data.content || data.error || JSON.stringify(data)));
    append("agentTrace", "Error", data.content || data.error || JSON.stringify(data));
  } else if (etype === "done") {
    setStatus("Done.");
    currentJobId = null;
    setRunning(false);
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  } else if (etype === "heartbeat") {
    // ignore
  } else if (etype === "graph_update") {
    // intentionally ignored for now
  } else {
    append("agentTrace", "Event: " + etype, JSON.stringify(data, null, 2));
  }
}

function bindEventSourceListeners(es) {
  const eventTypes = [
    "job_start",
    "run_start",
    "status",
    "next_speaker",
    "agent_message",
    "tool_call_message",
    "tool_call",
    "tool_result",
    "final_answer",
    "final",
    "cancelled",
    "error",
    "done",
    "heartbeat",
    "graph_update"
  ];

  for (const eventType of eventTypes) {
    es.addEventListener(eventType, function(event) {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        data = {
          type: eventType,
          content: event.data
        };
      }

      handleEvent(eventType, data);
    });
  }

  es.onerror = function() {
    // EventSource also triggers onerror when the server closes the stream.
    // If the job is already done, we ignore this.
    if (currentJobId) {
      append("agentTrace", "EventSource Error", "Connection interrupted or closed.");
      setStatus("Connection closed.");
      currentJobId = null;
      setRunning(false);
    }

    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  };
}

async function ask() {
  const question = document.getElementById("question").value.trim();
  if (!question) return;

  clearAll();
  document.getElementById("question").value = question;

  currentJobId = uuid();

  setRunning(true);
  setStatus("Running job " + currentJobId);
  setAnswer("Running SAGE-QA...\n\nJob ID: " + currentJobId);

  const url =
    "/qa_stream_get?question=" +
    encodeURIComponent(question) +
    "&job_id=" +
    encodeURIComponent(currentJobId);

  eventSource = new EventSource(url);
  bindEventSourceListeners(eventSource);
}

async function stopJob() {
  if (!currentJobId) {
    setStatus("No active job.");
    return;
  }

  const jobId = currentJobId;
  setStatus("Stopping job " + jobId + "...");

  try {
    await fetch("/qa_cancel/" + encodeURIComponent(jobId), {
      method: "POST"
    });
  } catch (err) {
    append("agentTrace", "Stop Error", err.toString());
  }

  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }

  currentJobId = null;
  setRunning(false);
  setAnswer("Stopped.");
  setStatus("Stopped.");
}

setRunning(false);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------

def create_fastapi_app(args: argparse.Namespace) -> FastAPI:
    global APP_ARGS
    APP_ARGS = normalize_args(args)

    worker_config = make_worker_config(APP_ARGS)

    app = FastAPI(
        title="SAGE-QA",
        description="Shared-Context Agentic Graph-Enhanced Question Answering",
        version="0.1.0",
    )

    @app.get("/")
    def root():
        return {
            "name": "SAGE-QA",
            "routes": {
                "ui": "/ui",
                "docs": "/docs",
                "health": "/health",
                "qa": "/qa",
                "qa_stream": "/qa_stream",
                "qa_stream_get": "/qa_stream_get",
                "qa_cancel": "/qa_cancel/{job_id}",
            },
        }

    @app.get("/ui", response_class=HTMLResponse)
    def ui():
        return HTMLResponse(UI_HTML)

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model": APP_ARGS.model,
            "base_url": APP_ARGS.base_url,
        }

    @app.post("/qa")
    def qa(req: QARequest):
        answer = ask_blocking(req.question)
        return {"answer": answer}

    def make_streaming_response(question: str, job_id: str):
        q = MP_CTX.Queue()
        proc = MP_CTX.Process(
            target=qa_stream_worker,
            args=(job_id, question, q, worker_config),
        )

        STREAM_JOBS[job_id] = {
            "process": proc,
            "queue": q,
            "created_at": now_ts(),
        }

        proc.start()

        def event_generator():
            yield sse_format({
                "type": "job_start",
                "agent": "system",
                "content": f"Started stream job {job_id}.",
                "metadata": {"job_id": job_id},
            })

            try:
                while True:
                    try:
                        event = q.get(timeout=0.5)
                        yield sse_format(event)

                        if event.get("type") == "done":
                            break

                    except queue.Empty:
                        if not proc.is_alive():
                            yield sse_format({
                                "type": "done",
                                "metadata": {"job_id": job_id},
                            })
                            break

                        yield ": heartbeat\n\n"

            finally:
                proc.join(timeout=1)

                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)

                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)

                STREAM_JOBS.pop(job_id, None)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.post("/qa_stream")
    def qa_stream(req: QAStreamRequest):
        job_id = req.job_id or uuid.uuid4().hex
        return make_streaming_response(req.question, job_id)

    @app.get("/qa_stream_get")
    def qa_stream_get(question: str, job_id: Optional[str] = None):
        job_id = job_id or uuid.uuid4().hex
        return make_streaming_response(question, job_id)

    @app.post("/qa_cancel/{job_id}")
    def qa_cancel(job_id: str):
        return terminate_stream_job(job_id)

    return app


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAGE-QA API + native Web UI server.")

    parser.add_argument(
        "--base-url",
        default=os.environ.get("QA_BASE_URL", "http://localhost:8080/v1"),
        help="OpenAI-compatible LLM endpoint base URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QA_MODEL", "llama3.3-70b"),
        help="Model id exposed by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("QA_DATA_DIR", "./GRAPHDATA_TSMC"),
        help="Graph data directory.",
    )
    parser.add_argument(
        "--data-dir-out",
        default=os.environ.get("QA_DATA_DIR_OUT", "./GRAPHDATA_TSMC_OUTPUT"),
        help="Graph output directory.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("QA_HOST", "0.0.0.0"),
        help="Host for SAGE-QA server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("QA_PORT", "8000")),
        help="Port for SAGE-QA server.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_fastapi_app(args)

    print("")
    print("SAGE-QA server starting")
    print(f"- UI:      http://{args.host}:{args.port}/ui")
    print(f"- Docs:    http://{args.host}:{args.port}/docs")
    print(f"- Health:  http://{args.host}:{args.port}/health")
    print(f"- QA:      POST http://{args.host}:{args.port}/qa")
    print(f"- Stream:  POST http://{args.host}:{args.port}/qa_stream")
    print(f"- Stream:  GET  http://{args.host}:{args.port}/qa_stream_get")
    print(f"- Stop:    POST http://{args.host}:{args.port}/qa_cancel/<job_id>")
    print("")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()