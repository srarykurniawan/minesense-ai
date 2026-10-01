import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("artifacts")
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(ROOT / "data" / "corpus", data / "corpus")
    os.environ.update({"MINESENSE_ARTIFACTS_DIR": str(tmp), "MINESENSE_DATA_DIR": str(data),
                       "MINESENSE_LLM_PROVIDER": "extractive", "MINESENSE_API_KEYS": "test-key",
                       "MINESENSE_RATE_LIMIT_PER_MINUTE": "1000"})
    from minesense.config import get_settings
    get_settings.cache_clear()
    from minesense.data.synth import write_all
    write_all(n_incidents=600, n_reports=60, n_wo=200)
    # hard set extraction ikut disalin
    shutil.copy(ROOT / "data" / "eval" / "shift_reports_hard.jsonl", data / "eval" / "shift_reports_hard.jsonl")
    yield


@pytest.fixture(scope="session")
def index(_env):
    from minesense.config import get_settings
    from minesense.rag.index import HybridIndex
    idx = HybridIndex(get_settings())
    idx.build_from_dir(get_settings().data_dir / "corpus")
    return idx


@pytest.fixture(scope="session")
def rag(index):
    from minesense.rag.pipeline import RAGPipeline
    return RAGPipeline(index)
