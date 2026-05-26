from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Tuple, Union

import networkx as nx
import pandas as pd
import torch
from autogen.agentchat.contrib.vectordb.base import Document
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from chromadb import PersistentClient
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from GraphReasoning import load_embeddings, save_embeddings, generate_node_embeddings

from config import SAGEConfig
from core import set_node_embeddings
from prompts import SYSTEM_PROMPT_GRAPHMAKER


def build_openai_client(config: SAGEConfig) -> OpenAI:
    return OpenAI(base_url=config.base_url, api_key=config.api_key)


def make_generate(config: SAGEConfig, client: OpenAI):
    import instructor

    def generate(system_prompt: str = "", prompt: str = "", temperature: float = 0, response_model=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if response_model is not None:
            create = instructor.patch(create=client.chat.completions.create)
            messages[0]["content"] = messages[0]["content"] + "\n" + SYSTEM_PROMPT_GRAPHMAKER
            return create(
                model=config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=None,
                response_model=response_model,
            )
        resp = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=None,
        )
        return resp.choices[0].message.content

    return generate


def load_embedding_model(config: SAGEConfig):
    if not config.embedding_model_path.exists():
        raise FileNotFoundError(
            f"Embedding model path not found: {config.embedding_model_path}\n"
            "Create a symlink such as: mkdir -p models && ln -s /actual/path/to/SEMIKONG-8b-GPTQ models/SEMIKONG-8b-GPTQ"
        )
    device_n = torch.cuda.device_count()
    device = f"cuda:{device_n-1}" if torch.cuda.is_available() and device_n > 0 else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(config.embedding_model_path), use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(config.embedding_model_path),
        device_map=device if device.startswith("cuda") else None,
        torch_dtype="auto",
    )
    if device == "cpu":
        model = model.to(device)
    return tokenizer, model, device


def load_graph_and_embeddings(config: SAGEConfig, tokenizer, model, device: str) -> nx.DiGraph:
    graph_path = config.data_dir_out / "tsmc_5b10p.graphml"
    if not graph_path.exists():
        raise FileNotFoundError(f"GraphML not found: {graph_path}")

    G = nx.read_graphml(str(graph_path))
    relation = nx.get_edge_attributes(G, "title")
    nx.set_edge_attributes(G, relation, "relation")
    nx.set_node_attributes(G, nx.pagerank(G), "pr")
    print(f"KG loaded: {G}")

    emb_path = config.data_dir / config.embedding_file
    regenerate = config.rebuild_embeddings or (not emb_path.exists())
    with torch.no_grad():
        if regenerate:
            print("Regenerating embeddings ...")
            embeddings = generate_node_embeddings(
                items=list(G.nodes),
                batch_size=32,
                tokenizer=tokenizer,
                model=model,
                device=device,
                pool="mean",
                normalize=False,
            )
            save_embeddings(embeddings, str(emb_path))
            print(f"Saved embeddings to: {emb_path}")
        else:
            print(f"Loading embeddings from: {emb_path}")
            embeddings = load_embeddings(str(emb_path))
    set_node_embeddings(embeddings)
    return G


def load_chunks(data_dir: Path) -> Tuple[List[str], List[str], List[str], List[Document]]:
    csv_files = sorted(glob.glob(str(data_dir / "*chunks_clean.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No *_chunks_clean.csv found in {data_dir}")

    chunk_ids: List[str] = []
    chunks: List[str] = []
    titles: List[str] = []
    lower_words = {"in", "to", "on", "for", "and", "or", "of", "with", "at", "by", "from", "as", "the", "a", "an"}

    for path in csv_files:
        df = pd.read_csv(path)
        base = path.replace("_chunks_clean.csv", "").split("/")[-1]
        parts = base.replace("_", " ").split(" ")[1:]
        parts_cap = []
        for idx, w in enumerate(parts):
            if not w:
                parts_cap.append("")
            elif idx == 0:
                parts_cap.append(w[:1].upper() + w[1:])
            elif w.lower() in lower_words:
                parts_cap.append(w.lower())
            else:
                parts_cap.append(w[:1].upper() + w[1:])
        title = " ".join(parts_cap)
        df["title"] = title
        chunk_ids += list(df["chunk_id"])
        chunks += list(df["text"])
        titles += list(df["title"])

    chunk_ids, chunks, titles = dedup_by_chunk_ids(chunk_ids, chunks, titles)
    docs = [Document(id=cid, content=c, metadata={"title": t}) for cid, c, t in zip(chunk_ids, chunks, titles)]
    return chunk_ids, chunks, titles, docs


def dedup_by_chunk_ids(chunk_ids, chunks, titles, keep="first"):
    if not (len(chunk_ids) == len(chunks) == len(titles)):
        raise ValueError("Lists must be the same length.")
    seen = set()
    keep_idx = []
    rng = range(len(chunk_ids)) if keep == "first" else range(len(chunk_ids) - 1, -1, -1)
    for i in rng:
        cid = chunk_ids[i]
        if cid in seen:
            continue
        seen.add(cid)
        keep_idx.append(i)
    keep_idx.sort()
    return [chunk_ids[i] for i in keep_idx], [chunks[i] for i in keep_idx], [titles[i] for i in keep_idx]


def make_embedding_function(tokenizer, model, device: str):
    def embedding_function(input: Union[str, List[str]], batch_size: int = 8, is_query: bool = False, **kwargs) -> List[List[float]]:
        import numpy as np
        if isinstance(input, (list, tuple)):
            texts = ["" if x is None else str(x).strip() for x in input]
        else:
            texts = ["" if input is None else str(input).strip()]
        texts = [t for t in texts if t]
        if not texts:
            return []
        embs = generate_node_embeddings(
            items=texts,
            batch_size=batch_size,
            tokenizer=tokenizer,
            model=model,
            device=device,
            embedding_function=None,
            pool="mean",
            normalize=True,
        )
        if isinstance(embs, dict):
            vecs = [np.asarray(embs[t], dtype=np.float32).tolist() for t in texts]
        else:
            arr = np.asarray(embs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            vecs = arr.tolist()
        return vecs
    return embedding_function


def setup_chroma(config: SAGEConfig, embedding_function, chunk_ids, chunks, titles):
    db_client = PersistentClient(path=str(config.chroma_dir))
    chroma_db = ChromaVectorDB(client=db_client, embedding_function=embedding_function)
    chroma_db.active_collection = db_client.get_or_create_collection(config.chroma_collection)
    chroma_db.active_collection._embed = embedding_function

    current_count = chroma_db.active_collection.count()
    if current_count < len(chunk_ids):
        if not config.rebuild_chroma and current_count > 0:
            print(f"Chroma collection has {current_count} docs; expected {len(chunk_ids)}. Upserting missing/all docs using notebook behavior.")
        elif not config.rebuild_chroma and current_count == 0:
            print("Chroma collection is empty. Upserting docs using notebook behavior.")
        chroma_db.active_collection.upsert(
            documents=chunks,
            ids=chunk_ids,
            metadatas=[{"title": title} for title in titles],
        )
    print(chroma_db.active_collection.count())
    return chroma_db
