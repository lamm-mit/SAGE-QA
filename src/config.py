from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class SAGEConfig:
    base_url: str = "http://localhost:8080/v1"
    model: str = "llama3.3-70b"
    api_key: str = "NULL"
    data_dir: Path = Path("./GRAPHDATA_TSMC")
    data_dir_out: Path = Path("./GRAPHDATA_TSMC_OUTPUT")
    embedding_model_path: Path = Path("./models/SEMIKONG-8b-GPTQ")
    embedding_file: str = "TSMC_SEMIKONG.pkl"
    chroma_dir: Path = Path("./chroma")
    chroma_collection: str = "TSMC_KG_SEMIKONG"
    cache_seed: int = 9527
    temperature: float = 0.0
    timeout: int = 10000
    max_tokens: int = 6000
    summarizer_max_tokens: int = 50000
    rebuild_embeddings: bool = False
    rebuild_chroma: bool = False
    debug: bool = False
    verbatim: bool = False

    @classmethod
    def from_env(cls) -> "SAGEConfig":
        return cls(
            base_url=os.getenv("QA_BASE_URL", "http://localhost:8080/v1"),
            model=os.getenv("QA_MODEL", "llama3.3-70b"),
            api_key=os.getenv("QA_API_KEY", "NULL"),
            data_dir=Path(os.getenv("QA_DATA_DIR", "./GRAPHDATA_TSMC")),
            data_dir_out=Path(os.getenv("QA_DATA_DIR_OUT", "./GRAPHDATA_TSMC_OUTPUT")),
            embedding_model_path=Path(os.getenv("QA_EMB_MODEL_PATH", "./models/SEMIKONG-8b-GPTQ")),
            embedding_file=os.getenv("QA_EMBEDDING_FILE", "TSMC_SEMIKONG.pkl"),
            chroma_dir=Path(os.getenv("QA_CHROMA_DIR", "./chroma")),
            chroma_collection=os.getenv("QA_CHROMA_COLLECTION", "TSMC_KG_SEMIKONG"),
            cache_seed=int(os.getenv("QA_CACHE_SEED", "9527")),
            temperature=float(os.getenv("QA_TEMPERATURE", "0")),
            timeout=int(os.getenv("QA_TIMEOUT", "10000")),
            max_tokens=int(os.getenv("QA_MAX_TOKENS", "6000")),
            summarizer_max_tokens=int(os.getenv("QA_SUMMARIZER_MAX_TOKENS", "50000")),
            rebuild_embeddings=os.getenv("QA_REBUILD_EMBEDDINGS", "0") == "1",
            rebuild_chroma=os.getenv("QA_REBUILD_CHROMA", "0") == "1",
            debug=os.getenv("QA_DEBUG", "0") == "1",
            verbatim=os.getenv("QA_VERBATIM", "0") == "1",
        )

    def resolved(self) -> "SAGEConfig":
        # Keep paths project-relative. Do not hard-code or resolve to workspace-specific absolute paths.
        self.data_dir = self.data_dir.expanduser()
        self.data_dir_out = self.data_dir_out.expanduser()
        self.embedding_model_path = self.embedding_model_path.expanduser()
        self.chroma_dir = self.chroma_dir.expanduser()
        return self

    @property
    def llm_config(self) -> dict:
        return {
            "cache_seed": self.cache_seed,
            "config_list": [self.config_list_item],
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
        }

    @property
    def llm_tool_config(self) -> dict:
        # Notebook used same config shape for normal/tool agents.
        return {
            "cache_seed": self.cache_seed,
            "config_list": [self.config_list_item],
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
        }

    @property
    def llm_config_summarizer(self) -> dict:
        return {
            "cache_seed": self.cache_seed,
            "config_list": [self.config_list_item],
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_tokens": self.summarizer_max_tokens,
        }

    @property
    def config_list_item(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_type": "openai",
            "api_key": self.api_key,
            "price": [0.0, 0.0],
        }
