from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel


class Node(BaseModel):
    id: str
    type: str


class Edge(BaseModel):
    source: str
    target: str
    relation: str
    metadata: Dict[str, str] = {}


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


class Nodes(BaseModel):
    nodes: List[Node]
