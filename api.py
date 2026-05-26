from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multiagent_qa.app import MultiAgentQAApp
from multiagent_qa.config import AppConfig
from multiagent_qa.deploy.runtime import set_runtime_env


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str


config = AppConfig.from_env(project_root=PROJECT_ROOT)
set_runtime_env(config)
qa_app = MultiAgentQAApp(config)
app = FastAPI(title="Multi-Agent GraphRAG QA API")


@app.on_event("startup")
def startup_event() -> None:
    qa_app.setup()


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> QAResponse:
    result = qa_app.ask(req.question, log_name="api_last")
    return QAResponse(answer=qa_app.final_answer(result))
