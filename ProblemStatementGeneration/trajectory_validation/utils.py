# io_schema_matcher.py
import re
import numpy as np
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 64
EPS = 1e-12



def split_at_capitals(word):
    return " ".join(re.findall(r'[A-Z][^A-Z]*', word))

def cosine(a, b):
    return cosine_similarity([a], [b])[0,0]

def split_param_name(name: str) -> str:
    """Convert camelCase/snake_case → space-separated lowercase."""
    if not name:
        return ""
    s = name.replace("_", " ")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", s)
    return " ".join(s.split()).lower()


def extract_type_and_desc(value: str) -> Tuple[str, str]:
    """'string (venue id)' → ('string', 'venue id')"""
    parts = value.strip().split()
    typ = parts[0].lower() if parts else ""
    desc = " ".join(parts[1:]) if len(parts) > 1 else ""
    return typ, desc


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================
# EMBEDDER
# =============================================
class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = BATCH_SIZE):
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)


# =============================================
# PARAMETER EMBEDDING
# =============================================
def embed_parameter(name: str, value: str, embedder: Embedder) -> Tuple[str, np.ndarray, str, np.ndarray]:
    name_text = split_param_name(name)
    typ, desc_text = extract_type_and_desc(value)

    name_emb = embedder.encode([name_text])[0] if name_text else np.zeros(embedder.dim, dtype=np.float32)
    desc_emb = embedder.encode([desc_text])[0] if desc_text else np.zeros(embedder.dim, dtype=np.float32)

    return name, name_emb, typ, desc_emb


# =============================================
# OUTPUT SCHEMA FLATTENING
# =============================================
def flatten_output_schema(schema: Any, embedder: Embedder, prefix: str = "") -> List[Tuple[str, np.ndarray, str, np.ndarray]]:
    outputs = []

    def walk(node: Any, path: str):
        if isinstance(node, dict):
            for k, v in node.items():
                new_path = f"{path}.{k}" if path else k
                walk(v, new_path)
        elif isinstance(node, list):
            if not node:
                outputs.append(embed_parameter(f"{path}[]", "list", embedder))
            else:
                elem = node[0]
                if isinstance(elem, (dict, list)):
                    walk(elem, f"{path}[0]")
                else:
                    typ = type(elem).__name__
                    outputs.append(embed_parameter(f"{path}[]", typ, embedder))
        else:
            typ = type(node).__name__ if not isinstance(node, str) else "string"
            outputs.append(embed_parameter(path, f"{typ} (leaf value)", embedder))

    if isinstance(schema, str):
        outputs.append(embed_parameter(prefix or "result", schema, embedder))
    else:
        walk(schema, prefix)

    return outputs


# =============================================
# API → EMBEDDINGS
# =============================================
def api_to_embeddings(api: Dict[str, Any], embedder: Embedder) -> Tuple[
    List[Tuple[str, np.ndarray, str, np.ndarray]],
    List[Tuple[str, np.ndarray, str, np.ndarray]]
]:
    inputs = []
    for pname, pval in api.get("params", {}).items():
        inputs.append(embed_parameter(pname, str(pval), embedder))

    outputs = flatten_output_schema(api.get("output", {}), embedder)
    return inputs, outputs