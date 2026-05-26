from __future__ import annotations

from typing import List, Union
from termcolor import colored

from core import collect_entities, shared_context, debug
from prompts import SYSTEM_PROMPT_GRAPHMAKER
from events import SAGEEvent, emit


def export_mind_graph_for_ui(max_nodes: int = 120, max_edges: int = 200):
    """Export the current shared mind map as frontend-safe node/edge JSON."""
    mind = shared_context.mind
    if not mind or not getattr(mind, "G", None):
        return {"nodes": [], "edges": []}

    graph = mind.G
    nodes = []
    edges = []

    try:
        for node, data in list(graph.nodes(data=True))[:max_nodes]:
            nodes.append({
                "id": str(node),
                "label": str(node),
                "data": {k: str(v) for k, v in dict(data).items()},
            })

        kept = {n["id"] for n in nodes}
        for u, v, data in list(graph.edges(data=True))[:max_edges]:
            if str(u) not in kept or str(v) not in kept:
                continue
            relation = data.get("relation") or data.get("label") or data.get("title") or ""
            edges.append({
                "source": str(u),
                "target": str(v),
                "label": str(relation),
                "data": {k: str(vv) for k, vv in dict(data).items()},
            })
    except Exception as exc:
        return {"nodes": [], "edges": [], "error": f"{type(exc).__name__}: {exc}"}

    return {"nodes": nodes, "edges": edges}


def formulate(query: str = "") -> str:
    """Ingest text into Thought-KG (MindMap). Notebook-faithful behavior."""
    if debug:
        print("FORMULATE")
    if not shared_context.mind:
        return "Context not initialized."
    shared_context.mind.intake(query, system_prompt=SYSTEM_PROMPT_GRAPHMAKER)
    return ""


def graph_source_rag(query: str = "", similarity_threshold: Union[float, str] = 0.95) -> str:
    """Notebook-faithful Graph Source RAG tool with structured event emission."""
    emit(SAGEEvent(
        type="tool_call",
        agent="engineer",
        tool="graph_source_rag",
        arguments={"query": query, "similarity_threshold": similarity_threshold},
    ))

    if not shared_context.knowledgebase:
        result = "Context not initialized."
        emit(SAGEEvent(type="tool_result", agent="user_proxy", tool="graph_source_rag", result=result))
        return result
    if debug:
        print("GRAPH_SOURCE_RAG")

    max_n = 1
    sim_thr = float(similarity_threshold)

    subgraph = shared_context.knowledgebase.extract_keywords_to_subgraph(query, max_n, sim_thr)
    paths_list = collect_entities(subgraph)
    paths = "\n".join(paths_list)

    if not paths_list:
        result = "No relations found. Abort."
        emit(SAGEEvent(type="tool_result", agent="user_proxy", tool="graph_source_rag", result=result))
        return result

    chunks: List[str] = []
    for _, _, data in subgraph.out_edges(data=True):
        cid = data.get("chunk_id")
        if cid:
            docs = shared_context.knowledgebase.chroma.get_docs_by_ids(cid)
            if docs:
                doc = docs[0]
                chunks.append(
                    f'Source chunk: {doc.get("content", doc.get("document"))} | title (DOI): {data.get("DOI", "")} | chunk_id: {cid}'
                )

    chunks_text = "\n".join(chunks)

    formulate(
        f"A group of specialized agents are working on this query: {query}\n\n. "
        f"Here is the information we found and need you to reorganize:\n"
        f"Retrived PATH:\n {paths}\n\nRetrieved Source Text:\n{chunks_text}"
    )

    print(colored(
        f"Aftering formulating, mind graph grows to {len(shared_context.mind.G.nodes)} nodes and "
        f"{len(shared_context.mind.G.edges)} edges {shared_context.mind.G if debug else ''}",
        "green",
    ))

    if debug:
        from datetime import datetime
        import networkx as nx
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        nx.write_graphml(shared_context.mind.G, f"mind_graph_{ts}.graphml")

    result = "\n".join(collect_entities(shared_context.mind.G))
    preview = result[:12000] + ("\n...[truncated for stream event]" if len(result) > 12000 else "")
    emit(SAGEEvent(
        type="tool_result",
        agent="user_proxy",
        tool="graph_source_rag",
        result=preview,
        metadata={
            "result_chars": len(result),
            "paths": len(paths_list),
            "chunks": len(chunks),
            "truncated": len(result) > 12000,
        },
    ))
    emit(SAGEEvent(
        type="graph_update",
        agent="engineer",
        tool="graph_source_rag",
        graph=export_mind_graph_for_ui(),
        metadata={
            "nodes": len(shared_context.mind.G.nodes),
            "edges": len(shared_context.mind.G.edges),
        },
    ))
    return result

