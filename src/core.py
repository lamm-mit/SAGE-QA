from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Protocol, Set, Tuple, Union, Any

import networkx as nx
import numpy as np

from schemas import Graph, Nodes

Vector = List[float]
EmbedFn = Callable[[List[str]], List[Vector]]
GenerateFn = Callable[..., str]

# These mirror notebook globals.
debug = False
max_keywords = 5
node_embeddings: Dict[str, np.ndarray] = {}


class ChromaCollectionLike(Protocol):
    def _embed(self, texts: Union[str, List[str]]) -> np.ndarray: ...


class ChromaLike(Protocol):
    def retrieve_docs(self, query: str, n_results: int, distance_threshold: float): ...
    def get_docs_by_ids(self, ids: Union[str, List[str]]) -> List[Mapping]: ...
    @property
    def active_collection(self) -> ChromaCollectionLike: ...


def set_debug(value: bool) -> None:
    global debug
    debug = bool(value)


def set_node_embeddings(value: Dict[str, np.ndarray]) -> None:
    global node_embeddings
    node_embeddings = value


def _as_nd(v: Union[Vector, np.ndarray]) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


def _l2_normalize(v: Union[Vector, np.ndarray], eps: float = 1e-12) -> np.ndarray:
    x = _as_nd(v)
    if x.ndim == 1:
        n = float(np.linalg.norm(x) + eps)
        return x / n
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n


def find_shortest_path_subgraph_between_nodes(graph: nx.DiGraph, nodes: List[Any]) -> nx.DiGraph:
    T: List[Any] = []
    seen = set()
    print(f"Looking for connections between {nodes}")
    for x in nodes:
        if x not in seen:
            T.append(x)
            seen.add(x)

    if not T:
        print("Path found ratio = 1.00 (empty)")
        return graph.subgraph([]).copy()
    if len(T) == 1:
        print("Path found ratio = 1.00 (single terminal)")
        return graph.subgraph([T[0]]).copy()

    pair_path: Dict[Tuple[Any, Any], List[Any]] = {}
    pair_len: Dict[Tuple[Any, Any], int] = {}
    all_pairs = 0
    found = 0
    for i, a in enumerate(T):
        for j, b in enumerate(T):
            if i == j:
                continue
            all_pairs += 1
            try:
                p = nx.shortest_path(graph, a, b)
                pair_path[(a, b)] = p
                pair_len[(a, b)] = len(p) - 1
                found += 1
            except nx.NetworkXNoPath:
                pass
    ratio = found / all_pairs if all_pairs else 1.0
    print(f"Path found ratio = {found}/{all_pairs} = {ratio:.2%}")

    succs: Dict[Any, List[Any]] = {u: [] for u in T}
    indeg: Dict[Any, int] = {u: 0 for u in T}
    for (u, v), _ in pair_len.items():
        succs[u].append(v)
    for u in T:
        for v in succs[u]:
            indeg[v] += 1

    uncovered = set(T)
    terminal_chains: List[List[Any]] = []
    while uncovered:
        starts = sorted(uncovered, key=lambda x: (indeg[x], T.index(x)))
        cur = starts[0]
        chain = [cur]
        uncovered.remove(cur)
        while True:
            candidates = [v for v in succs.get(cur, []) if v in uncovered]
            if not candidates:
                break
            nxt = min(candidates, key=lambda v: pair_len[(cur, v)])
            chain.append(nxt)
            uncovered.remove(nxt)
            cur = nxt
        terminal_chains.append(chain)

    keep_nodes = set()
    for chain in terminal_chains:
        for k in range(len(chain) - 1):
            u, v = chain[k], chain[k + 1]
            path = pair_path[(u, v)]
            if k == 0:
                keep_nodes.update(path)
            else:
                keep_nodes.update(path[1:])
    for chain in terminal_chains:
        if len(chain) == 1:
            keep_nodes.add(chain[0])
    return graph.subgraph(list(keep_nodes)).copy()


def collect_entities(graph: nx.DiGraph, chunk_ids: Optional[List[str]] = None) -> List[str]:
    want = set(chunk_ids) if chunk_ids is not None else None
    lines: List[str] = []
    for u, v, data in graph.out_edges(data=True):
        if debug:
            print(u, v, data)
        if want is not None:
            cid = data.get("chunk_id", "")
            if cid not in want:
                continue
        relation = data.get("relation")
        chunk_id = data.get("chunk_id")
        DOI = data.get("DOI")
        rel_txt = f"-[{relation}]->" if relation else "-->"
        line = f"{u} {rel_txt} {v}."
        line += f" | title (DOI): {DOI}" if DOI else ""
        line += f" | chunk_id: {chunk_id}" if chunk_id else ""
        lines.append(line)
        if debug:
            print(lines)
    return lines


def euclidean_distance(a: Union[Vector, np.ndarray], b: Union[Vector, np.ndarray]) -> float:
    return float(np.linalg.norm(_as_nd(a).ravel() - _as_nd(b).ravel()))


class KnowledgeBase:
    def __init__(
        self,
        graph: Optional[nx.DiGraph],
        embed_fn: EmbedFn,
        *,
        chroma: Optional[ChromaLike] = None,
        generate: Optional[GenerateFn] = None,
        node_embeddings: Optional[Dict[str, np.ndarray]] = None,
        sim_merge: float = 0.92,
        sim_query: float = 0.90,
        top_k: int = 3,
        node_text: Optional[Callable[[str, Mapping], str]] = None,
    ):
        self.G = graph if graph is not None else nx.DiGraph()
        self.embed_fn = embed_fn
        self.generate = generate
        self.chroma = chroma
        self.node_embeddings = node_embeddings or {}
        self.sim_merge = float(sim_merge)
        self.sim_query = float(sim_query)
        self.top_k = int(top_k)
        self.node_text = node_text or (lambda nid, data: str(nid))

        if node_embeddings:
            missing = [n for n in self.G.nodes if n not in self.node_embeddings]
            if missing:
                self._embed_nodes(missing)

    def _embed_nodes(self, node_ids: List[str]) -> None:
        texts = [self.node_text(nid, self.G.nodes[nid]) for nid in node_ids]
        vecs = self.embed_fn(texts)
        for nid, v in zip(node_ids, vecs):
            self.node_embeddings[nid] = _l2_normalize(v)

    def _embed_text(self, text: str) -> np.ndarray:
        return _l2_normalize(self.embed_fn([text])[0])

    def similar_nodes(self, keyword: str, top_k: Optional[int] = None, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        if not self.node_embeddings:
            return []
        top_k = self.top_k if top_k is None else top_k
        threshold = self.sim_query if threshold is None else threshold
        keyword_embedding = self.embed_fn(keyword)[0]

        import heapq
        from scipy.spatial.distance import cosine
        min_heap = []
        heapq.heapify(min_heap)
        for node, embedding in self.node_embeddings.items():
            embedding = embedding.flatten()
            similarity = 1 - cosine(keyword_embedding, embedding)
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (similarity, node))
            elif similarity > min_heap[0][0]:
                heapq.heappop(min_heap)
                heapq.heappush(min_heap, (similarity, node))
        best_nodes = sorted(min_heap, key=lambda x: -x[0])
        # Preserve notebook behavior: always keep best node; threshold applies to remaining nodes.
        filtered = []
        for idx, (similarity, node) in enumerate(best_nodes):
            if idx == 0 or similarity >= threshold:
                filtered.append((node, similarity))
        return filtered

    def extract_keywords_to_subgraph(self, query, max_n_samples, similarity_threshold, aggressive=False):
        keywords = self.extract_keywords(query)
        print(f"Found keywords: {keywords} in {query}")
        return self.keywords_to_subgraph(keywords, max_n_samples, similarity_threshold, aggressive=False)

    def keywords_to_subgraph(self, keywords: List[str], max_n_samples: int = 3, similarity_threshold: float = 0.9, aggressive: bool = False) -> nx.DiGraph:
        chosen: Set[str] = set()
        for kw in keywords:
            cands = self.similar_nodes(kw, top_k=max_n_samples, threshold=similarity_threshold)
            if not cands:
                continue
            if aggressive and len(cands) > 1:
                cands = sorted(cands, key=lambda x: (self.G.degree[x[0]], x[1]), reverse=True)
            for cand in cands:
                chosen.add(cand[0])
            if debug:
                print(f"cands: {cands} found in keywords_to_subgraph")
        if debug:
            print(f"chosen: {chosen} found in keywords_to_subgraph")
        if not chosen:
            return self.G.subgraph([]).copy()
        if len(chosen) == 1:
            node = list(chosen)[0]
            graph_nodes = {node}
            graph_nodes.update(self.G.successors(node))
            graph_nodes.update(self.G.predecessors(node))
            if debug:
                print(f"single node: {node}, expanded with neighbors: {graph_nodes}")
            return self.G.subgraph(graph_nodes).copy()
        graph = find_shortest_path_subgraph_between_nodes(self.G, list(chosen))
        return self.G.subgraph(graph).copy()

    def extract_keywords(self, query: str) -> List[str]:
        if not self.generate:
            raise RuntimeError("generate is not set on KnowledgeBase")
        resp = self.generate(
            system_prompt=f"""Identify the keywords, each of which is in one to three words, and can partially overlap with each other, in the context.
            Never give zero, one, or more than {max_keywords} keywords. Expand the context if the question is too generic.""",
            prompt=f"Context: ```{query}```",
            response_model=Nodes,
        )
        keywords = [n.id for n in resp.nodes]
        if debug:
            print(f"{keywords} found in extract_keywords_to_subgraph")
        return keywords


class MindMap(KnowledgeBase):
    def __init__(self, graph: Optional[nx.DiGraph] = None, embed_fn: Optional[EmbedFn] = None, generate: Optional[GenerateFn] = None, **kw):
        super().__init__(graph or nx.DiGraph(), embed_fn=embed_fn or (lambda xs: [np.zeros(1, dtype=np.float32)] * len(xs)), generate=generate, **kw)

    def intake(self, content: str, system_prompt: str) -> None:
        if not self.generate:
            raise RuntimeError("generate is not set on MindMap")
        resp = self.generate(system_prompt=system_prompt, prompt=f"Text:\n```{content}```", response_model=Graph)
        newG = nx.DiGraph()
        for node in resp.nodes:
            newG.add_node(node.id, type=node.type)
        for edge in resp.edges:
            newG.add_edge(edge.source, edge.target, relation=edge.relation)
        self.G = nx.compose(self.G, newG)

    def refresh(self) -> None:
        self.G = nx.DiGraph()


@dataclass
class Context:
    mind: Optional[MindMap] = None
    knowledgebase: Optional[KnowledgeBase] = None
    chroma: Optional[ChromaLike] = None
    generate: Optional[GenerateFn] = None
    verbatim: bool = False


shared_context = Context()


def init_context(*, graph: Optional[nx.DiGraph], embed_fn: EmbedFn, chroma: Optional[ChromaLike], generate: GenerateFn, system_prompt_graph_maker: str = "", verbatim: bool = False) -> None:
    shared_context.mind = MindMap(nx.DiGraph(), embed_fn=embed_fn, generate=generate)
    shared_context.knowledgebase = KnowledgeBase(graph, embed_fn=embed_fn, node_embeddings=node_embeddings, chroma=chroma, generate=generate)
    shared_context.chroma = chroma
    shared_context.generate = generate
    shared_context.verbatim = bool(verbatim)
