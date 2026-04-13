from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List, Tuple, Optional, Dict
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

def get_all_collections(client: QdrantClient) -> List[str]:
    cols = client.get_collections()
    return [c.name for c in cols.collections]


def _cosine(a, b) -> float:
    va = np.array(a); vb = np.array(b)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def _normalize_score(raw_score: float, distance_mode: str) -> float:
    """
    统一到“越大越好”的相似度分数：
      - cosine / dot: 认为原始分数已是相似度 → 原样返回
      - euclid:       原始分数是距离 → 相似度 = 1/(1+d)
    """
    dm = (distance_mode or "cosine").lower()
    if dm in ("euclid", "euclidean", "l2"):
        d = max(float(raw_score), 0.0)
        return 1.0 / (1.0 + d)
    return float(raw_score)

def search_top_k_global(
    client: QdrantClient,
    collections: List[str],
    embeddings: HuggingFaceEmbeddings,
    query: str,
    k: int = 5,
    score_threshold: Optional[float] = None,  # euclid下表示距离阈值（越小越严格）
    per_collection_fetch: Optional[int] = None,
    # ------- 重排加权 -------
    collection_boosts: Optional[Dict[str, float]] = None,  # 例: {"long_term_memory": 0.35}
    title_sim_weight: float = 0.30,                        # 标题相似度权重 [0,1]
    combine_mode: str = "mul",                             # "mul"（稳）或 "add"（激进）
    # ------- 全局距离模式 -------
    distance_mode: str = "cosine",                         # "cosine" | "dot" | "euclid"
) -> List[Tuple[float, str, str, str, str]]:
    """
    返回：(final_score, collection, point_id, title, text)，按融合分数降序。
    融合：
      base_sim  = normalize(score, distance_mode)
      title_sim = cosine(query_vec, title_vec)
      col_boost = collection_boosts.get(col, 0)

      mul: final = base_sim * (1+col_boost) * (1 + title_sim_weight * title_sim)
      add: final = base_sim + col_boost + title_sim_weight * title_sim
    """
    if not collections:
        return []

    per_k = per_collection_fetch if per_collection_fetch else max(k, 10)
    qvec = embeddings.embed_query(query)

    collection_boosts = collection_boosts or {}
    title_vec_cache: Dict[str, List[float]] = {}
    candidates: List[Tuple[float, str, str, str, str]] = []

    for col in collections:
        kwargs = dict(
            collection_name=col,
            query_vector=qvec,
            limit=per_k,
            with_payload=True,
        )
        if score_threshold is not None:
            # 注意：若 distance_mode="euclid"，这是“距离上限”阈值
            kwargs["score_threshold"] = score_threshold

        results = client.search(**kwargs)

        for pt in results:
            payload = pt.payload or {}
            title = payload.get("title", "无标题")
            text = payload.get("text", "[无内容]")
            pid = getattr(pt, "id", None)
            raw_score = float(getattr(pt, "score", 0.0))

            base_sim = _normalize_score(raw_score, distance_mode)

            # 标题相似度（用同一模型）
            if title in title_vec_cache:
                tvec = title_vec_cache[title]
            else:
                tvec = embeddings.embed_query(title)
                title_vec_cache[title] = tvec
            title_sim = _cosine(qvec, tvec)

            col_boost = float(collection_boosts.get(col, 0.0))

            if combine_mode == "add":
                final_score = base_sim + col_boost + title_sim_weight * title_sim
            else:  # 默认 "mul"
                final_score = base_sim * (1.0 + col_boost) * (1.0 + title_sim_weight * title_sim)

            candidates.append((final_score, col, str(pid), title, text))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:k]


def search_top_k_global_ltm_priority(
    client: QdrantClient,
    collections: List[str],
    embeddings: HuggingFaceEmbeddings,
    query: str,
    k: int = 5,
    score_threshold: Optional[float] = None,     # euclid 下是距离上限；不确定可用 None
    per_collection_fetch: Optional[int] = None,  # 每库抓取候选条数（合并后再取Top-K）
    *,
    # 只对 long_term_collection 生效的加权
    long_term_collection: str = "long_term_memory",
    collection_boosts: Optional[Dict[str, float]] = None,   # 例: {"long_term_memory": 0.35}
    title_sim_weight: float = 0.30,                        # 仅 long_term_collection 使用
    combine_mode: str = "mul",                             # "mul"（稳）或 "add"（激进）
    distance_mode: str = "cosine",                         # 全局距离模式: "cosine"|"dot"|"euclid"
) -> List[Tuple[float, str, str, str, str]]:
    """
    仅对 long_term_collection 应用标题权重与集合权重；其他集合只用基于 text 的分数。

    返回：(final_score, collection, point_id, title, text)，按 final_score 降序。
    """
    if not collections:
        return []

    per_k = per_collection_fetch if per_collection_fetch else max(k, 10)
    qvec = embeddings.embed_query(query)

    collection_boosts = collection_boosts or {}
    title_vec_cache: Dict[str, List[float]] = {}
    candidates: List[Tuple[float, str, str, str, str]] = []

    for col in collections:
        kwargs = dict(
            collection_name=col,
            query_vector=qvec,
            limit=per_k,
            with_payload=True,
        )
        if score_threshold is not None:
            kwargs["score_threshold"] = score_threshold

        results = client.search(**kwargs)

        # 仅 long_term_collection 走“标题权重 + 集合权重”
        is_ltm = (col == long_term_collection)
        ltm_boost = float(collection_boosts.get(col, 0.0)) if is_ltm else 0.0

        for pt in results:
            payload = pt.payload or {}
            title = payload.get("title", "无标题")
            text = payload.get("text", "[无内容]")
            pid = getattr(pt, "id", None)
            raw_score = float(getattr(pt, "score", 0.0))

            base_sim = _normalize_score(raw_score, distance_mode)

            if is_ltm:
                # 标题相似度（统一用余弦，范围[-1,1]，稳定好调）
                if title in title_vec_cache:
                    tvec = title_vec_cache[title]
                else:
                    tvec = embeddings.embed_query(title)
                    title_vec_cache[title] = tvec
                title_sim = _cosine(qvec, tvec)

                if combine_mode == "add":
                    final_score = base_sim + ltm_boost + title_sim_weight * title_sim
                else:  # "mul"
                    final_score = base_sim * (1.0 + ltm_boost) * (1.0 + title_sim_weight * title_sim)
            else:
                # 其他集合：只用文本的基础分（不加标题、不加集合权重）
                final_score = base_sim

            candidates.append((final_score, col, str(pid), title, text))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:k]



def pretty_print_results(results: List[Tuple[float, str, str, str, str]]):
    if not results:
        print("未检索到结果。")
        return
    print("\n===== 全局 Top-K 检索结果 =====\n")
    for i, (score, col, pid, title, text) in enumerate(results, 1):
        preview = (text or "").replace("\n", " ")[:120]
        print(f"{i}. score={score:.6f} | collection={col} | id={pid}")
        print(f"   title：{title}")
        print(f"   preview：{preview}...\n")

if __name__ == "__main__":

    # ---------- 必要参数 ----------
    QDRANT_PATH = "../Qdrant_Data/all-mpnet-base-v2_cosine_v1.0"
    EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

    # ---------- 初始化 ----------
    qdrant = QdrantClient(path=QDRANT_PATH)
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    all_cols = get_all_collections(qdrant)

    # query = "ISO/IEC 17825"
    # query = "Timing Analysis"
    query = "Simple Power Analysis"
    # query = "Differential Power Analysis"

    # query = "Perform Simple Power Analysis (SPA) on the RSA algorithm trace collected from a single-chip microcomputer. The trace file is located at: ./trace_data/RSA_STM32F429/RSA-STM32F429-9030points.npy, and the trace can be divided into 36 segments."
    # query = "Help me detect that the AES cryptographic algorithm implemented by a smart card conforms to ISO/IEC 17825, and its operation clock frequency is 4800Hz. The TA trace with a 5M sampling rate is: ./trace_data/AES_7816/AES_7816_TA_fixed&random_plain_5M_1000traces_220000points.npy, The TVLA trace with a 5M sampling rate is: ./trace_data/AES_7816/AES_7816_TVLA_fixed&random_5M_40000traces_4187points.npy, The names of trace information files are also given. Help me check whether the implementation is safe by ISO/IEC 17825."

    top_k = 5
    distance_metric = "cosine"
    # distance_metric = "dot"
    # distance_metric = "euclid"

    # results = search_top_k_global(
    #     client=qdrant,
    #     collections=all_cols,
    #     embeddings=emb,
    #     query=query,
    #     k=top_k,
    #     distance_mode=distance_metric,
    #     collection_boosts={"long_term_memory": 0.45},
    #     title_sim_weight=0.4,
    #     combine_mode="mul",     #add 更激进, 可能把低基础分的条目拉上来
    # )

    results = search_top_k_global_ltm_priority(
        client=qdrant,
        collections=all_cols,
        embeddings=emb,
        query=query,
        k=top_k,
        distance_mode=distance_metric,
        long_term_collection="long_term_memory",
        collection_boosts={"long_term_memory": 0.70},  # 只对 long_term_memory 生效
        title_sim_weight=0.80,                         # 只对 long_term_memory 生效
        combine_mode="mul",
    )

    pretty_print_results(results)
