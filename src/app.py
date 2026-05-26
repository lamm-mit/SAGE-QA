from __future__ import annotations

import os
from pathlib import Path

# Must be set before importing autogen in normal runtime.
os.environ.setdefault("AUTOGEN_USE_RESPONSES_API", "0")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CHROMADB_MAX_BATCH_SIZE", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32")

import torch

from config import SAGEConfig
from deploy.vllm_patch import apply_vllm_patch
from deploy.runtime import (
    build_openai_client,
    load_embedding_model,
    load_graph_and_embeddings,
    load_chunks,
    make_embedding_function,
    setup_chroma,
    make_generate,
)
from core import init_context, set_debug, shared_context
from agents.factory import build_agents, build_groupchat, reset_all_agents, clear_cache_dir, save_answer_outputs
from sample_questions import DEFAULT_QUESTIONS
from events import EventEmitter, SAGEEvent, set_current_emitter


class SAGEQAApp:
    def __init__(self, config: SAGEConfig):
        self.config = config.resolved()
        self.ready = False
        self.client = None
        self.generate = None
        self.embedding_tokenizer = None
        self.embedding_model = None
        self.embedding_device = None
        self.graph = None
        self.chroma = None
        self.agents = None
        self.user_proxy = None
        self.planner = None
        self.engineer = None
        self.critic = None
        self.summarizer = None
        self.groupchat = None
        self.manager = None

    def setup(self) -> None:
        apply_vllm_patch()
        set_debug(self.config.debug)
        clear_cache_dir()

        self.client = build_openai_client(self.config)
        self.generate = make_generate(self.config, self.client)

        self.embedding_tokenizer, self.embedding_model, self.embedding_device = load_embedding_model(self.config)
        self.graph = load_graph_and_embeddings(
            self.config,
            self.embedding_tokenizer,
            self.embedding_model,
            self.embedding_device,
        )
        chunk_ids, chunks, titles, _docs = load_chunks(self.config.data_dir)
        embedding_function = make_embedding_function(
            self.embedding_tokenizer,
            self.embedding_model,
            self.embedding_device,
        )
        self.chroma = setup_chroma(self.config, embedding_function, chunk_ids, chunks, titles)

        init_context(
            graph=self.graph,
            embed_fn=embedding_function,
            chroma=self.chroma,
            generate=self.generate,
            verbatim=self.config.verbatim,
        )

        self.user_proxy, self.planner, self.engineer, self.critic, self.summarizer = build_agents(self.config)
        self.agents, self.groupchat, self.manager = build_groupchat(
            self.user_proxy,
            self.planner,
            self.engineer,
            self.critic,
            self.summarizer,
        )
        self.ready = True

    def check_endpoint(self) -> str:
        if self.client is None:
            self.client = build_openai_client(self.config)
        models = self.client.models.list()
        return str(models)

    def ask(self, question: str, output_name: str = "manual"):
        if not self.ready:
            self.setup()
        with torch.no_grad():
            if shared_context.mind:
                shared_context.mind.refresh()
            reset_all_agents([self.planner, self.engineer, self.critic, self.summarizer, self.user_proxy], self.manager)
            answer = self.user_proxy.initiate_chat(self.manager, message=f"{question}\n")
            save_answer_outputs(Path("outputs"), output_name, question, answer)
            return answer

    def ask_stream(self, question: str, emitter: EventEmitter, output_name: str = "stream"):
        """Run QA while emitting structured visible runtime events.

        This preserves the normal ask() behavior and only adds event emission.
        """
        if not self.ready:
            self.setup()

        set_current_emitter(emitter)
        try:
            emitter.emit(SAGEEvent(type="run_start", agent="user", content=question))
            answer = self.ask(question, output_name=output_name)
            emitter.emit(SAGEEvent(type="final_answer", agent="summarizer", content=str(answer)))
            return answer
        finally:
            set_current_emitter(None)

    def ask_sample(self, sample_index: int):
        if sample_index < 1 or sample_index > len(DEFAULT_QUESTIONS):
            raise ValueError(f"sample_index must be 1..{len(DEFAULT_QUESTIONS)}")
        question = DEFAULT_QUESTIONS[sample_index - 1]
        return self.ask(question, output_name=f"q{sample_index}")
