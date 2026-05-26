# SAGE-QA

**SAGE-QA**: **Shared-Context Agentic Graph-Enhanced Question Answering**.

SAGE-QA is a graph-grounded multi-agent question-answering framework for scientific and engineering reasoning over technical corpora. The current implementation targets semiconductor manufacturing QA, where answers often depend on connected evidence across mechanisms, materials, process conditions, equipment variables, figures, tables, and source documents.

This version keeps the original notebook-style AutoGen behavior while making the code runnable as a small Python project. The directory name is `sage_qa`, and the Python source files live directly under `src/` without an extra `src/sage_qa/` package layer.

---

## Overview

SAGE-QA is designed around the idea that expert QA should not only retrieve relevant text, but also preserve the relationships among technical concepts.

The system combines:

- graph-grounded retrieval
- source-linked relation paths
- shared evolving evidence state
- role-specialized multi-agent reasoning
- critic-guided refinement
- final technical synthesis

The high-level workflow is:

```text
User question
   ↓
Planner
   ↓
Engineer
   ↓
Graph/source RAG tool
   ↓
Shared mind map
   ↓
Critic
   ↓
Planner refinement
   ↓
Summarizer
   ↓
Final answer
```

Only the **Engineer** agent calls the retrieval tool. Planner, Critic, and Summarizer operate through natural-language reasoning over retrieved evidence and intermediate answers.

---

## Directory Layout

Recommended project layout:

```text
sage_qa/
├── main.py
├── serve.py
├── api.py
├── requirements.txt
├── pyproject.toml
├── README.md
├── agent_settings/
│   ├── common_rules.md
│   ├── planner.md
│   ├── engineer.md
│   ├── critic.md
│   ├── summarizer.md
│   ├── user_proxy.md
│   └── graphmaker.md
├── src/
│   ├── app.py
│   ├── config.py
│   ├── core.py
│   ├── prompts.py
│   ├── sample_questions.py
│   ├── schemas.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── factory.py
│   ├── knowledge/
│   │   ├── __init__.py
│   │   └── tools.py
│   └── deploy/
│       ├── __init__.py
│       ├── runtime.py
│       └── vllm_patch.py
├── models/
├── chroma/
├── GRAPHDATA_TSMC/
├── GRAPHDATA_TSMC_OUTPUT/
└── outputs/
```

Runtime data should stay local and should not be committed:

```text
models/
chroma/
GRAPHDATA_TSMC/
GRAPHDATA_TSMC_OUTPUT/
outputs/
```

The runtime data folders can be real folders or symbolic links to external locations. For example:

```text
chroma -> /path/to/chroma
GRAPHDATA_TSMC -> /path/to/graph/data
GRAPHDATA_TSMC_OUTPUT -> /path/to/graph/output
models -> /path/to/local/models
```

You can also manually specify the graph data locations at runtime with `--data-dir` and `--data-dir-out`.

---

## Required LLM Endpoint

`main.py` and `serve.py` do not start the LLM model. You need to first host or connect to an OpenAI-compatible endpoint, for example a local vLLM server:

```text
http://localhost:8080/v1
```

Check the endpoint:

```bash
curl http://localhost:8080/v1/models
```

Use the returned model id as `--model`.

Example vLLM launch:

```bash
vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8080 \
  --served-model-name llama3.3-70b
```

In this setup:

```text
SAGE-QA API server:
http://localhost:8000

LLM inference server:
http://localhost:8080/v1
```

The SAGE-QA server calls the LLM inference server internally.

---

## Environment Setup

```bash
cd sage_qa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use an existing conda environment:

```bash
conda activate llm
pip install -r requirements.txt
```

---

## Local Data Setup

Expected graph data:

```text
GRAPHDATA_TSMC/
├── TSMC_SEMIKONG.pkl
└── *_chunks_clean.csv

GRAPHDATA_TSMC_OUTPUT/
└── tsmc_5b10p.graphml
```

Expected Chroma cache:

```text
chroma/
```

Expected local embedding model path:

```text
models/SEMIKONG-8b-GPTQ
```

Recommended symlink setup:

```bash
mkdir -p models
ln -s /actual/path/to/SEMIKONG-8b-GPTQ models/SEMIKONG-8b-GPTQ
ln -s /actual/path/to/chroma chroma
ln -s /actual/path/to/GRAPHDATA_TSMC GRAPHDATA_TSMC
ln -s /actual/path/to/GRAPHDATA_TSMC_OUTPUT GRAPHDATA_TSMC_OUTPUT
```

If you do not want to use the default relative paths, manually pass the data paths at runtime:

```bash
python main.py \
  --base-url http://localhost:8080/v1 \
  --model llama3.3-70b \
  --data-dir /path/to/GRAPHDATA_TSMC \
  --data-dir-out /path/to/GRAPHDATA_TSMC_OUTPUT \
  --sample-index 1
```

For server mode:

```bash
python serve.py \
  --base-url http://localhost:8080/v1 \
  --model llama3.3-70b \
  --data-dir /path/to/GRAPHDATA_TSMC \
  --data-dir-out /path/to/GRAPHDATA_TSMC_OUTPUT \
  --port 8000
```

If `--data-dir` and `--data-dir-out` are not provided, SAGE-QA expects:

```text
./GRAPHDATA_TSMC
./GRAPHDATA_TSMC_OUTPUT
```

---

## Run One-Shot QA

Use `main.py` for manual testing.

Sample Q1:

```bash
python main.py \
  --base-url http://localhost:8080/v1 \
  --model llama3.3-70b \
  --data-dir ./GRAPHDATA_TSMC \
  --data-dir-out ./GRAPHDATA_TSMC_OUTPUT \
  --sample-index 1
```

Custom question:

```bash
python main.py \
  --base-url http://localhost:8080/v1 \
  --model llama3.3-70b \
  --data-dir ./GRAPHDATA_TSMC \
  --data-dir-out ./GRAPHDATA_TSMC_OUTPUT \
  -q "What are the knobs that can change the uniformity in radical Si-etching process?"
```

Endpoint check only:

```bash
python main.py \
  --check \
  --base-url http://localhost:8080/v1 \
  --model llama3.3-70b
```

Outputs are written to:

```text
outputs/
```

---

## Serve API Mode

Start the SAGE-QA API server:

```bash
python serve.py \
  --base-url http://localhost:8080/v1 \
  --model llama3.3-70b \
  --data-dir ./GRAPHDATA_TSMC \
  --data-dir-out ./GRAPHDATA_TSMC_OUTPUT \
  --port 8000
```

Call the QA endpoint:

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the knobs that can change the uniformity in radical Si-etching process?"}'
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## Environment Variables

You can also set the LLM endpoint and model with environment variables:

```bash
export QA_BASE_URL=http://localhost:8080/v1
export QA_MODEL=llama3.3-70b
```

Then run:

```bash
python main.py \
  --data-dir ./GRAPHDATA_TSMC \
  --data-dir-out ./GRAPHDATA_TSMC_OUTPUT \
  --sample-index 1
```

or:

```bash
python serve.py \
  --data-dir ./GRAPHDATA_TSMC \
  --data-dir-out ./GRAPHDATA_TSMC_OUTPUT \
  --port 8000
```

---

## Agent Roles

### Planner

Planner decomposes the user query into retrieval-friendly sub-questions and controls the reasoning roadmap.

### Engineer

Engineer is the only agent allowed to call the graph/source RAG tool. It retrieves evidence and writes grounded intermediate answers.

### Critic

Critic evaluates Engineer's intermediate answer for correctness, coverage, and clarity.

### Summarizer

Summarizer produces the final answer using Planner's structure and Engineer's grounded evidence.

---

## Core Retrieval Function

The main retrieval function is:

```text
graph_source_rag(query, similarity_threshold)
```

It performs:

1. query keyword or phrase extraction
2. entity matching against the domain knowledge graph
3. shortest-path retrieval
4. source chunk resolution through Chroma
5. shared mind-map update
6. return of graph-grounded context

Conceptually:

```text
query
   ↓
matched graph entities
   ↓
shortest relation paths
   ↓
linked source chunks
   ↓
shared mind map
   ↓
agent-readable context
```

---

## Paper

This repository implements the SAGE-QA workflow described in the manuscript:

```bibtex
@article{hsu2026sageqa,
  title   = {SAGE-QA: Shared-Context Agentic Graph-Enhanced Question Answering for Scientific Reasoning in Semiconductor Manufacturing Technology},
  author  = {Hsu, Yu-Chuan and Buehler, Markus J.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## Citation

If you use this codebase, please cite:

```bibtex
@article{hsu2026sageqa,
  title   = {SAGE-QA: Shared-Context Agentic Graph-Enhanced Question Answering for Scientific Reasoning in Semiconductor Manufacturing Technology},
  author  = {Hsu, Yu-Chuan and Buehler, Markus J.},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

This project is released under the MIT License.

```text
MIT License

Copyright (c) 2026 Yu-Chuan Hsu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights    
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     
copies of the Software, and to permit persons to whom the Software is        
furnished to do so, subject to the following conditions:                     

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.                              

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

---

## Acknowledgment

This project was developed for research on graph-grounded multi-agent scientific question answering in semiconductor manufacturing technology.
