import re
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re
from typing import List
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# 工具
# =========================
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
    统一到“越大越好”的相似度：
      - cosine/dot: 直接使用原始分
      - euclid:     原始是距离 → 相似度 = 1/(1+d)
    """
    dm = (distance_mode or "cosine").lower()
    if dm in ("euclid", "euclidean", "l2"):
        d = max(float(raw_score), 0.0)
        return 1.0 / (1.0 + d)
    return float(raw_score)

def infer_distance_mode_from_path(db_path: str, default: str = "cosine") -> str:
    s = (db_path or "").lower()
    if re.search(r"(?:^|[_\-./])cosine(?:$|[_\-./])", s): return "cosine"
    if re.search(r"(?:^|[_\-./])dot(?:$|[_\-./])", s):    return "dot"
    if re.search(r"(?:^|[_\-./])(euclid|euclidean|l2)(?:$|[_\-./])", s): return "euclid"
    return default

def _sanitize(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (s or '').lower())

# =========================
# 检索（固定：LTM 优先）
# =========================
def search_top_k_global_ltm_priority(
    client: QdrantClient,
    collections: List[str],
    embeddings: HuggingFaceEmbeddings,
    query: str,
    *,
    k: int = 5,
    long_term_collection: str = "long_term_memory",
    collection_boost: float = 0.35,   # 仅对 LTM 生效
    title_sim_weight: float = 0.40,   # 仅对 LTM 生效
    combine_mode: str = "mul",        # 固定乘法融合（更稳）
    distance_mode: str = "cosine",
    per_collection_fetch: Optional[int] = None,
    score_threshold: Optional[float] = None,  # euclid 下为“最大距离”；不确定就 None
) -> List[Tuple[float, str, str, str, str]]:
    """
    返回：(final_score, collection, point_id, title, text)，按 final_score 降序。
    仅 LTM 使用标题相似度与集合加权；其他集合仅用 base_sim。
    """
    if not collections:
        return []

    per_k = per_collection_fetch if per_collection_fetch else max(k, 10)
    qvec = embeddings.embed_query(query)

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
        is_ltm = (col == long_term_collection)

        for pt in results:
            payload = pt.payload or {}
            title = payload.get("title", "无标题")
            text  = payload.get("text", "[无内容]")
            pid   = getattr(pt, "id", None)
            raw_score = float(getattr(pt, "score", 0.0))

            base_sim = _normalize_score(raw_score, distance_mode)

            if is_ltm:
                # 仅 LTM 计算标题相似度并融合
                if title in title_vec_cache:
                    tvec = title_vec_cache[title]
                else:
                    tvec = embeddings.embed_query(title)
                    title_vec_cache[title] = tvec
                title_sim = _cosine(qvec, tvec)

                if combine_mode == "add":
                    final_score = base_sim + collection_boost + title_sim_weight * title_sim
                else:  # "mul"
                    final_score = base_sim * (1.0 + collection_boost) * (1.0 + title_sim_weight * title_sim)
            else:
                final_score = base_sim

            candidates.append((final_score, col, str(pid), title, text))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:k]

# =========================
# 相关性与评分：Q@k / nDCG@k
# =========================
def _is_relevant(query: str, title: str) -> bool:
    # 现阶段相关性：标题是否包含查询（已做规范化）
    qk = _sanitize(query)
    return bool(qk) and (qk in _sanitize(title))

def compute_q_at_k(
    query: str,
    results: List[Tuple[float, str, str, str, str]],  # (score, collection, id, title, text)
    *,
    k: Optional[int] = None,
    ltm_name: str = "long_term_memory",
) -> Tuple[float, str, List[Dict[str, Any]]]:
    """Q@k：LTM&相关=10，非LTM&相关=5，否则0；输出 0~1、百分比、明细"""
    if not results:
        return 0.0, "0.00%", []
    items = results[: (k or len(results))]

    total_gain = 0.0
    details = []
    for rank, (_, col, pid, title, text) in enumerate(items, 1):
        rel = _is_relevant(query, title)
        if rel and col == ltm_name:
            g = 10.0
        elif rel and col != ltm_name:
            g = 5.0
        else:
            g = 0.0
        total_gain += g
        details.append({
            "rank": rank, "id": pid, "collection": col, "title": title,
            "gain": g, "relevant": bool(rel)
        })

    Q01 = total_gain / (10.0 * len(items))
    return Q01, f"{Q01*100:.2f}%", details

def compute_ndcg_k(
    query: str,
    results: List[Tuple[float, str, str, str, str]],
    *,
    k: Optional[int] = None,
    ltm_name: str = "long_term_memory",
) -> Tuple[float, str]:
    """nDCG@k（基于同样的 10/5/0 收益，输出 0~1、百分比）"""
    if not results:
        return 0.0, "0.00%"
    items = results[: (k or len(results))]

    def gain(col: str, title: str) -> float:
        if _is_relevant(query, title) and col == ltm_name:
            return 10.0
        if _is_relevant(query, title):
            return 5.0
        return 0.0

    gains = [gain(col, title) for _, col, _, title, _ in items]
    dcg = sum(g / math.log2(i+2) for i, g in enumerate(gains))

    # IDCG：同一批 items 的收益从大到小重排
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i+2) for i, g in enumerate(ideal))
    ndcg = 0.0 if idcg == 0 else dcg / idcg
    return ndcg, f"{ndcg*100:.2f}%"

# =========================
# 包装：搜索 + 打分（统一为 Q@k + nDCG@k）
# =========================
def search_and_score(
    *,
    client: QdrantClient,
    collections: List[str],
    embeddings: HuggingFaceEmbeddings,
    query: str,
    db_path: Optional[str] = None,
    top_k: int = 5,
    long_term_collection: str = "long_term_memory",
    # 检索阶段参数
    collection_boost: float = 0.35,
    title_sim_weight: float = 0.80,
    combine_mode: str = "mul",
    per_collection_fetch: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> Tuple[
    List[Tuple[float, str, str, str, str]],
    float, str,     # Q@k
    float, str,     # nDCG@k
    List[Dict[str, Any]]  # 明细
]:
    distance_mode = infer_distance_mode_from_path(db_path or "", default="cosine")

    results = search_top_k_global_ltm_priority(
        client=client,
        collections=collections,
        embeddings=embeddings,
        query=query,
        k=top_k,
        long_term_collection=long_term_collection,
        collection_boost=collection_boost,
        title_sim_weight=title_sim_weight,
        combine_mode=combine_mode,
        distance_mode=distance_mode,
        per_collection_fetch=per_collection_fetch,
        score_threshold=score_threshold,
    )

    Q01, Qpct, details = compute_q_at_k(
        query=query,
        results=results,
        k=top_k,
        ltm_name=long_term_collection,
    )
    n01, npct = compute_ndcg_k(
        query=query,
        results=results,
        k=top_k,
        ltm_name=long_term_collection,
    )
    return results, Q01, Qpct, n01, npct, details

# =========================
# 输出工具
# =========================
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

# =========================
# 示例
# =========================
# if __name__ == "__main__":
#     QDRANT_PATH = "../Qdrant_Data/all-mpnet-base-v2_cosine_v1.0"
#     qdrant = QdrantClient(path=QDRANT_PATH)
#
#     # 建议开启归一化，余弦/点积更稳定
#     emb = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-mpnet-base-v2",
#         encode_kwargs={"normalize_embeddings": True}
#     )
#
#     all_cols = get_all_collections(qdrant)
#
#     queries = [
#         "ISO/IEC 17825",
#         # "Timing Analysis",
#         # "Simple Power Analysis",
#         # "Differential Power Analysis",
#     ]
#
#     for query in queries:
#         print("\n" + "=" * 80)
#         print(f"🔎 Query: {query}")
#
#         results, Q01, Qpct, n01, npct, details = search_and_score(
#             client=qdrant,
#             collections=all_cols,
#             embeddings=emb,
#             query=query,
#             db_path=QDRANT_PATH,
#             top_k=5,
#             long_term_collection="long_term_memory",
#             collection_boost=0.70,
#             title_sim_weight=0.80,
#             combine_mode="mul",
#             per_collection_fetch=None,
#             score_threshold=None,
#         )
#
#         # pretty_print_results(results)
#
#         # print("=== Evaluation ===")
#         print(f"Q@5   : {Qpct}")
#         print(f"nDCG@5: {npct}")
#
#         # # 如需查看每条 gain/相关性：
#         # for d in details:
#         #     print(d)



ROOT_DIR = "../Qdrant_Data"

EMBED_MAP = [
    (r"all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2"),
    (r"bge-base-en-v1\.5", "BAAI/bge-base-en-v1.5"),
    (r"gte-base-en-v1\.5", "Alibaba-NLP/gte-base-en-v1.5"),
]

def pick_embed_id(dir_name: str) -> str:
    for pat, mid in EMBED_MAP:
        if re.search(pat, dir_name):
            return mid
    return "sentence-transformers/all-mpnet-base-v2"

def list_vector_dbs(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    dirs = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            if re.search(r"(mpnet|bge|gte).*(cosine|dot|euclid)", name, re.I):
                dirs.append(full)
    dirs.sort()
    return dirs

if __name__ == "__main__":
    queries = [
        "ISO/IEC 17825",
        # "Timing Analysis",
        # "Simple Power Analysis",
        # "Differential Power Analysis",
    ]

    vector_dbs = list_vector_dbs(ROOT_DIR)
    if not vector_dbs:
        print(f"未在 {ROOT_DIR} 下发现向量库目录。")
        raise SystemExit(0)

    # 可视化检查顺序，确认包含 cosine 那个库
    print("将要评测的库顺序：", [os.path.basename(p) for p in vector_dbs])

    for db_path in vector_dbs:
        dir_name = os.path.basename(db_path)
        embed_id = pick_embed_id(dir_name)
        norm = not re.search(r"(euclid|euclidean|l2)", dir_name, re.I)

        print("\n" + "=" * 100)
        print(f"📦 DB: {dir_name}")
        print(f"   • Embedding: {embed_id}")
        print(f"   • normalize_embeddings: {norm}")

        try:
            qdrant = QdrantClient(path=db_path)

            # 是否需要 trust_remote_code（gte 系列通常需要）
            need_trust = bool(re.search(r"gte", embed_id, re.I))
            emb = HuggingFaceEmbeddings(
                model_name=embed_id,
                encode_kwargs={"normalize_embeddings": norm},
                model_kwargs={"trust_remote_code": need_trust}
            )

            all_cols = get_all_collections(qdrant)
            if not all_cols:
                print("   ⚠️ 该库没有任何 collection，跳过。")
                continue

            for query in queries:
                results, Q01, Qpct, n01, npct, details = search_and_score(
                    client=qdrant,
                    collections=all_cols,
                    embeddings=emb,
                    query=query,
                    db_path=db_path,                  # 自动推断距离模式
                    top_k=5,
                    long_term_collection="long_term_memory",
                    collection_boost=0.3,
                    title_sim_weight=0.2,
                    combine_mode="mul",
                    per_collection_fetch=None,
                    score_threshold=None,
                )
                print(f"\n🔎 Query: {query}")
                print(f"   Q@5   : {Qpct}")
                print(f"   nDCG@5: {npct}")

                # 如需展开每条结果：
                # pretty_print_results(results)
                # for d in details: print("   ", d)

        except Exception as e:
            print(f"   ❌ 评测失败：{e}")
