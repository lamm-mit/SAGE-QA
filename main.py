#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import SAGEConfig
from app import SAGEQAApp


def parse_args():
    p = argparse.ArgumentParser(description="Run SAGE-QA one-shot QA from terminal.")
    p.add_argument("-q", "--question", default=None, help="Question to ask.")
    p.add_argument("--sample-index", type=int, default=None, help="Run sample question by 1-based index.")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible LLM endpoint, e.g. http://localhost:8080/v1")
    p.add_argument("--model", default=None, help="Served model id from /v1/models")
    p.add_argument("--api-key", default=None, help="API key. For local vLLM usually NULL.")
    p.add_argument("--data-dir", default=None, help="Graph/chunk data dir, e.g. ./GRAPHDATA_TSMC")
    p.add_argument("--data-dir-out", default=None, help="Graph output dir, e.g. ./GRAPHDATA_TSMC_OUTPUT")
    p.add_argument("--embedding-model-path", default=None, help="Local embedding model path. Default ./models/SEMIKONG-8b-GPTQ")
    p.add_argument("--chroma-dir", default=None, help="Chroma persistent dir. Default ./chroma")
    p.add_argument("--check", action="store_true", help="Only check LLM endpoint.")
    p.add_argument("--rebuild-embeddings", action="store_true", help="Regenerate node embeddings pkl.")
    p.add_argument("--rebuild-chroma", action="store_true", help="Force Chroma rebuild/upsert behavior.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SAGEConfig.from_env()
    if args.base_url is not None:
        cfg.base_url = args.base_url
    if args.model is not None:
        cfg.model = args.model
    if args.api_key is not None:
        cfg.api_key = args.api_key
    if args.data_dir is not None:
        cfg.data_dir = Path(args.data_dir)
    if args.data_dir_out is not None:
        cfg.data_dir_out = Path(args.data_dir_out)
    if args.embedding_model_path is not None:
        cfg.embedding_model_path = Path(args.embedding_model_path)
    if args.chroma_dir is not None:
        cfg.chroma_dir = Path(args.chroma_dir)
    if args.rebuild_embeddings:
        cfg.rebuild_embeddings = True
    if args.rebuild_chroma:
        cfg.rebuild_chroma = True
    if args.debug:
        cfg.debug = True

    app = SAGEQAApp(cfg)

    if args.check:
        print(app.check_endpoint())
        return

    if args.sample_index is None and not args.question:
        raise SystemExit("Provide --sample-index N or -q/--question.")
    if args.sample_index is not None:
        answer = app.ask_sample(args.sample_index)
    else:
        answer = app.ask(args.question, output_name="manual")
    print(answer)


if __name__ == "__main__":
    main()
