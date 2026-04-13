# This script slices RAG documents with the selected embedding model and distance, then stores vectors into the matching local Qdrant database.
# Default embedding model: sentence-transformers/all-mpnet-base-v2; adjust as needed.

import os
import json
import chardet
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
from uuid import uuid4
from datetime import datetime
from sklearn.manifold import TSNE

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, PointStruct
import shutil


DISTANCE_NAME_MAP = {
    "cosine": "Cosine",
    "cos": "Cosine",
    "dot": "Dot",
    "dot_product": "Dot",
    "euclidean": "Euclid",
    "euclid": "Euclid",
    "l2": "Euclid",
}


def normalize_distance(distance: str) -> str:
    key = distance.lower()
    if key not in DISTANCE_NAME_MAP:
        valid = ', '.join(sorted(set(DISTANCE_NAME_MAP.keys())))
        raise ValueError(f"distance must be one of {valid}, got: {distance}")
    return DISTANCE_NAME_MAP[key]


# ================================
# 💾 数据存入 Qdrant 函数区域
# ================================


def store_structured_json_to_qdrant(
    json_path: str,
    qdrant_client,
    embeddings,
    collection_name: str,
    distance: str,
):
    """
    Vectorize structured KB entries where each entry is:
      { "title": "...", "steps": [ {"step": "...", "detail": "..."}, ... ] }
    For each entry, all steps are merged into ONE document so it becomes ONE vector
    (unless it exceeds chunk_size and must be split).
    """
    import os, json, chardet
    from uuid import uuid4
    from langchain_text_splitters import CharacterTextSplitter
    from qdrant_client.models import VectorParams, PointStruct

    distance_name = normalize_distance(distance)

    # 1) Load JSON with detected encoding
    with open(json_path, "rb") as f:
        encoding = chardet.detect(f.read(10000)).get("encoding") or "utf-8"

    try:
        with open(json_path, "r", encoding=encoding) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[ERROR] Failed to parse JSON: {exc}")
        return

    if not isinstance(data, list):
        print("[ERROR] JSON root must be a list of entries.")
        return

    # 2) Splitter: large chunk to avoid splitting; no overlap
    splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0, separator="\n")

    # 3) Build documents per entry (merge all steps into one text)
    docs = []
    src_name = os.path.basename(json_path)

    for item in data:
        title = item.get("title", "[No Title]")
        steps = item.get("steps")

        if not isinstance(steps, list) or not steps:
            print(f"[ERROR] Entry '{title}' has no 'steps' array; skipped.")
            continue

        step_lines = []
        for s in steps:
            step_name = s.get("step", "").strip()
            detail = s.get("detail", "").strip()
            line = f"{step_name}: {detail}" if step_name else detail
            step_lines.append(line)

        full_text = f"{title}\n\n" + "\n".join(step_lines)

        metadata = {
            "title": title,
            "source": src_name,
        }

        chunk_docs = splitter.create_documents([full_text], metadatas=[metadata])
        docs.extend(chunk_docs)

    if not docs:
        print(f"[WARN] No documents produced from {json_path}; aborting.")
        return

    # 4) Embeddings
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)
    if not vectors:
        print(f"[ERROR] Vector generation failed for {json_path}.")
        return
    dim = len(vectors[0])

    # 5) Create/replace collection
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=distance_name),
    )

    # 6) Upload
    points = []
    for i, d in enumerate(docs):
        payload = dict(d.metadata)
        payload["text"] = d.page_content
        points.append(PointStruct(id=str(uuid4()), vector=vectors[i], payload=payload))

    qdrant_client.upload_points(collection_name=collection_name, points=points)
    print(f"Stored {len(points)} vectors to Qdrant collection: {collection_name}")



def store_each_txt_to_db(txt_path, qdrant_client, embeddings, distance):
    """Ingest each TXT file under the given directory into a dedicated Qdrant collection."""
    if not os.path.isdir(txt_path):
        print(f"TXT path does not exist: {txt_path}")
        return

    distance_name = normalize_distance(distance)
    txt_files = [f for f in os.listdir(txt_path) if f.endswith(".txt")]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")

    for txt_file in txt_files:
        collection_name = os.path.splitext(txt_file)[0]
        file_path = os.path.join(txt_path, txt_file)

        with open(file_path, "rb") as f:
            encoding = chardet.detect(f.read(10000))["encoding"] or "utf-8"

        loader = TextLoader(file_path, encoding=encoding)
        documents = loader.load()
        chunks = splitter.split_documents(documents)

        texts = [doc.page_content for doc in chunks]
        if not texts:
            print(f"No text chunks were produced from {txt_file}, skipping")
            continue

        vectors = embeddings.embed_documents(texts)
        if not vectors:
            print(f"Vector generation failed for {txt_file}, skipping")
            continue

        payloads = []
        for doc in chunks:
            doc.metadata["source"] = txt_file
            doc.metadata["text"] = doc.page_content
            payloads.append(doc.metadata)

        vector_dim = len(vectors[0])

        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=distance_name),
        )

        points = [
            PointStruct(id=str(uuid4()), vector=vectors[i], payload=payloads[i])
            for i in range(len(vectors))
        ]
        qdrant_client.upload_points(collection_name=collection_name, points=points)
        print(f"Stored TXT file {txt_file} into Qdrant collection: {collection_name}")

def store_single_txt_to_qdrant(txt_path, qdrant_client, embeddings, collection_name, distance):
    assert txt_path.endswith(".txt"), "path must point to a .txt file"

    distance_name = normalize_distance(distance)

    with open(txt_path, "rb") as f:
        encoding = chardet.detect(f.read(10000))["encoding"] or "utf-8"

    loader = TextLoader(txt_path, encoding=encoding)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    chunks = splitter.split_documents(documents)

    texts = [doc.page_content for doc in chunks]
    if not texts:
        print(f"No text chunks were produced from {txt_path}, skipping")
        return

    vectors = embeddings.embed_documents(texts)
    if not vectors:
        print(f"Vector generation failed for {txt_path}, skipping")
        return

    title = os.path.splitext(os.path.basename(txt_path))[0]
    source = os.path.basename(txt_path)

    payloads = []
    for doc in chunks:
        payloads.append({
            "title": title,
            "source": source,
            "text": doc.page_content,
        })

    vector_dim = len(vectors[0])

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=distance_name),
    )

    points = [
        PointStruct(id=str(uuid4()), vector=vectors[i], payload=payloads[i])
        for i in range(len(vectors))
    ]
    qdrant_client.upload_points(collection_name=collection_name, points=points)
    print(f"Stored TXT into Qdrant collection: {collection_name}")



def store_each_pdf_to_db(pdf_path, qdrant_client, embeddings, distance):
    """Ingest each PDF file under the given directory into a dedicated Qdrant collection."""
    if not os.path.isdir(pdf_path):
        print(f"PDF path does not exist: {pdf_path}")
        return

    distance_name = normalize_distance(distance)
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")

    for pdf_file in pdf_files:
        collection_name = os.path.splitext(pdf_file)[0]
        file_path = os.path.join(pdf_path, pdf_file)

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        chunks = splitter.split_documents(documents)

        texts = [doc.page_content for doc in chunks]
        if not texts:
            print(f"No text chunks were produced from {pdf_file}, skipping")
            continue

        vectors = embeddings.embed_documents(texts)
        if not vectors:
            print(f"Vector generation failed for {pdf_file}, skipping")
            continue

        # 新的 payload 格式
        title = os.path.splitext(pdf_file)[0]
        source = pdf_file
        payloads = []
        for doc in chunks:
            payloads.append({
                "title": title,
                "source": source,
                "text": doc.page_content,
            })

        vector_dim = len(vectors[0])

        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=distance_name),
        )

        points = [
            PointStruct(id=str(uuid4()), vector=vectors[i], payload=payloads[i])
            for i in range(len(vectors))
        ]
        qdrant_client.upload_points(collection_name=collection_name, points=points)
        print(f"Stored PDF file {pdf_file} into Qdrant collection: {collection_name}")


def store_single_pdf_to_qdrant(pdf_path, qdrant_client, embeddings, collection_name, distance):
    assert pdf_path.endswith(".pdf"), "path must point to a .pdf file"

    distance_name = normalize_distance(distance)

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
    chunks = splitter.split_documents(documents)

    texts = [doc.page_content for doc in chunks]
    if not texts:
        print(f"No text chunks were produced from {pdf_path}, skipping")
        return

    vectors = embeddings.embed_documents(texts)
    if not vectors:
        print(f"Vector generation failed for {pdf_path}, skipping")
        return

    title = os.path.splitext(os.path.basename(pdf_path))[0]
    source = os.path.basename(pdf_path)

    payloads = []
    for doc in chunks:
        payloads.append({
            "title": title,
            "source": source,
            "text": doc.page_content,
        })

    vector_dim = len(vectors[0])

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=distance_name),
    )

    points = [
        PointStruct(id=str(uuid4()), vector=vectors[i], payload=payloads[i])
        for i in range(len(vectors))
    ]
    qdrant_client.upload_points(collection_name=collection_name, points=points)
    print(f"Stored PDF into Qdrant collection: {collection_name}")


def inspect_qdrant_collections_detailed(qdrant_client, preview_limit=5):
    """
    列出所有 Qdrant collection，打印每个集合的向量数量和前几个向量的详细 payload。
    """
    collections = qdrant_client.get_collections().collections
    if not collections:
        print("⚠️ 当前没有任何 Collection。")
        return

    print(f"\n📦 共找到 {len(collections)} 个 Collection：\n")

    for col in collections:
        name = col.name
        count = qdrant_client.count(collection_name=name, exact=True).count
        print(f"🔹 Collection: {name} ➤ 向量数量: {count}")

        result = qdrant_client.scroll(
            collection_name=name,
            limit=preview_limit,
            with_payload=True,
            with_vectors=False
        )
        points = result[0]
        for i, pt in enumerate(points, 1):
            print(f"   ➤ 向量 {i}:")
            print(f"      ID: {pt.id}")
            if pt.payload:
                for key, value in pt.payload.items():
                    if key == "text":
                        text_preview = (value[:100] + "...") if isinstance(value, str) and len(value) > 100 else value
                        print(f"      text: {text_preview}")
                    else:
                        print(f"      {key}: {value}")
            else:
                print("      ⚠️ 没有 payload")
        print("-" * 60)


from typing import Optional, Dict, Any, List
import math

def inspect_qdrant_collection(
    qdrant_client,
    collection_name: str,
    preview_limit: int = 5,
    show_vectors: bool = False,
) -> None:
    """
    查看单个 Qdrant collection 的配置信息与示例数据。

    Args:
        qdrant_client: QdrantClient 实例
        collection_name: 要查看的集合名
        preview_limit: 预览的点数量
        show_vectors: 是否展示向量信息（仅显示长度与前几维，避免刷屏）
    """
    # ---- 1) 基本信息 ----
    try:
        info = qdrant_client.get_collection(collection_name=collection_name)
    except Exception as e:
        print(f"❌ 获取集合信息失败: {e}")
        return

    try:
        total = qdrant_client.count(collection_name=collection_name, exact=True).count
    except Exception:
        total = None

    print(f"\n🔎 Inspect Collection: {collection_name}")
    print("—" * 60)

    # 状态、分片、副本等（不同版本字段可能略有差异，尽量兼容访问）
    status = getattr(info, "status", None)
    cfg = getattr(info, "config", None)
    params = getattr(cfg, "params", None) if cfg else None

    replication_factor = getattr(params, "replication_factor", None) if params else None
    write_consistency_factor = getattr(params, "write_consistency_factor", None) if params else None
    shard_number = getattr(params, "shard_number", None) if params else None

    print(f"• Status: {status}")
    if total is not None:
        print(f"• Vectors (count): {total}")
    if shard_number is not None:
        print(f"• Shards: {shard_number}")
    if replication_factor is not None:
        print(f"• Replication factor: {replication_factor}")
    if write_consistency_factor is not None:
        print(f"• Write consistency: {write_consistency_factor}")

    # ---- 2) 向量配置（默认向量 / 命名向量） ----
    def _extract_vector_configs(collection_info) -> List[Dict[str, Any]]:
        out = []
        _cfg = getattr(collection_info, "config", None)
        _params = getattr(_cfg, "params", None) if _cfg else None
        if not _params:
            return out

        # 可能是 params.vectors 或 params.vectors_config；可能是对象或 map
        cand = getattr(_params, "vectors", None)
        cand2 = getattr(_params, "vectors_config", None)

        def _push(name: str, vobj: Any):
            size = getattr(vobj, "size", None)
            dist = getattr(vobj, "distance", None)
            out.append({
                "name": name,
                "size": size,
                "distance": str(dist) if dist is not None else None
            })

        if cand is not None:
            if isinstance(cand, dict):  # 命名向量
                for name, v in cand.items():
                    _push(name, v)
            else:  # 单向量
                _push("default", cand)
        elif cand2 is not None:
            if isinstance(cand2, dict):
                for name, v in cand2.items():
                    _push(name, v)
            else:
                _push("default", cand2)
        return out

    vconfs = _extract_vector_configs(info)
    if vconfs:
        print("• Vector configs:")
        for vc in vconfs:
            print(f"  - name={vc['name']}, size={vc['size']}, distance={vc['distance']}")
    else:
        print("• Vector configs: (not available)")

    # 量化与 HNSW（有些版本在 config 上，有些在 params 上）
    quant = getattr(cfg, "quantization_config", None) if cfg else None
    hnsw = getattr(cfg, "hnsw_config", None) if cfg else None
    if quant:
        print("• Quantization: available")
    if hnsw:
        m = getattr(hnsw, "m", None)
        efc = getattr(hnsw, "ef_construct", None)
        print(f"• HNSW: m={m}, ef_construct={efc}")

    print("—" * 60)

    # ---- 3) 示例点预览 ----
    try:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=max(1, preview_limit),
            with_payload=True,
            with_vectors=show_vectors,
        )
        points = result[0] if isinstance(result, (list, tuple)) else []
    except Exception as e:
        print(f"❌ Scroll 失败: {e}")
        points = []

    if not points:
        print("⚠️ 无示例点（集合为空或无权限）。")
        return

    for i, pt in enumerate(points, 1):
        print(f"   ➤ Point {i}:")
        print(f"      ID: {getattr(pt, 'id', None)}")

        payload = getattr(pt, "payload", None) or {}
        if payload:
            # 优先打印常见字段
            title = payload.get("title")
            if title is not None:
                print(f"      title: {title}")
            source = payload.get("source")
            if source is not None:
                print(f"      source: {source}")

            # 文本预览
            text = payload.get("text")
            if isinstance(text, str):
                preview = text.replace("\n", " ")
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                print(f"      text: {preview}")

            # 其余键
            for k, v in payload.items():
                if k in {"title", "source", "text"}:
                    continue
                print(f"      {k}: {v}")
        else:
            print("      ⚠️ 无 payload")

        if show_vectors:
            vecs = getattr(pt, "vector", None) or getattr(pt, "vectors", None)
            if isinstance(vecs, dict):  # 命名向量
                for name, arr in vecs.items():
                    length = len(arr) if hasattr(arr, "__len__") else None
                    head = list(arr[:5]) if hasattr(arr, "__getitem__") else []
                    print(f"      vector[{name}]: len={length}, head={head}")
            elif vecs is not None:
                length = len(vecs) if hasattr(vecs, "__len__") else None
                head = list(vecs[:5]) if hasattr(vecs, "__getitem__") else []
                print(f"      vector: len={length}, head={head}")

    print("—" * 60)


def delete_qdrant_collection(qdrant_client, collection_name: str, confirm: bool = True):
    if not qdrant_client.collection_exists(collection_name):
        print(f"⚠️ Collection '{collection_name}' 不存在。")
        return

    if confirm:
        user_input = input(f"⚠️ 确定删除 Collection '{collection_name}'？（输入 y 确认）: ")
        if user_input.lower() != "y":
            print("❎ 已取消删除。")
            return

    qdrant_client.delete_collection(collection_name)
    print(f"✅ Collection '{collection_name}' 已删除。")

    # 强制删除残留文件（仅限本地模式）
    if hasattr(qdrant_client, "config") and hasattr(qdrant_client.config, "path"):
        col_path = os.path.join(qdrant_client.config.path, "collections", collection_name)
        if os.path.exists(col_path):
            shutil.rmtree(col_path)
            print(f"🗑️ 本地文件夹 '{col_path}' 已清理。")


def visualize_multiple_collections_vectors(qdrant_client, collection_names, method="tsne", limit_per_collection=300):
    all_vectors, all_labels, all_colors = [], [], []
    color_map = plt.cm.get_cmap('tab10', len(collection_names))

    for idx, name in enumerate(collection_names):
        print(f"📥 正在读取 Collection：{name}")
        result = qdrant_client.scroll(name, limit=limit_per_collection, with_vectors=True, with_payload=True)
        points = result[0]
        if not points:
            continue
        vectors = [pt.vector for pt in points]
        all_vectors.extend(vectors)
        all_labels.extend([name] * len(vectors))
        all_colors.extend([color_map(idx)] * len(vectors))

    if not all_vectors:
        print("❌ 无法可视化，所有 collection 都为空。")
        return

    reduced = TSNE(n_components=2).fit_transform(np.array(all_vectors)) if method == "tsne" \
        else umap.UMAP(n_components=2).fit_transform(np.array(all_vectors))

    plt.figure(figsize=(12, 7))
    for i, name in enumerate(collection_names):
        indices = [j for j, lbl in enumerate(all_labels) if lbl == name]
        coords = reduced[indices]
        plt.scatter(coords[:, 0], coords[:, 1], label=name, alpha=0.7, edgecolors='k')

    plt.title(f"向量分布图（{method.upper()} 降维）")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_database_vector_stats(qdrant_client):
    """
    检查已有数据库中所有 Collection 的向量数量，并返回总计。
    """
    try:
        # 获取所有 Collection 列表
        collections_response = qdrant_client.get_collections()
        collections = collections_response.collections

        if not collections:
            print("\n⚠️  当前数据库中没有任何 Collection。")
            return 0

        print(f"\n{'=' * 50}")
        print(f"{'Collection 名称':<30} | {'向量数量':>10}")
        print(f"{'-' * 50}")

        total_vectors = 0
        for col in collections:
            name = col.name
            # 使用 exact=True 获取物理上的精确计数
            count_result = qdrant_client.count(
                collection_name=name,
                exact=True
            )
            count = count_result.count
            total_vectors += count
            print(f"{name:<30} | {count:>10}")

        print(f"{'-' * 50}")
        print(f"{'总计 (Total)':<30} | {total_vectors:>10}")
        print(f"{'=' * 50}\n")

        return total_vectors

    except Exception as e:
        print(f"❌ 统计过程出错: {e}")
        return 0





# ================================
# 🧪 主入口
# ================================


if __name__ == "__main__":
    # --- 1. 配置已有数据库路径 (请确保此处路径与您之前生成的数据库一致) ---
    model_name = "sentence-transformers/all-mpnet-base-v2"
    distance_metric = "Cosine"
    version_tag = "v1.0"  # 使用您之前定义的版本标签

    model_alias = model_name.split("/")[-1]
    # 数据库存放的目录名
    db_folder_name = f"{model_alias}_{distance_metric.lower()}_{version_tag}"
    qdrant_path = os.path.join("..", "Qdrant_Data", db_folder_name)

    # --- 2. 检查目录是否存在 ---
    if not os.path.exists(qdrant_path):
        print(f"❌ 错误: 找不到数据库目录 '{qdrant_path}'。")
        print("请检查路径设置是否正确，或是否已经生成过该数据库。")
    else:
        print(f"✅ 正在连接至已有数据库: {qdrant_path}")

        # 初始化客户端（不执行任何写入操作）
        qdrant_client = QdrantClient(path=qdrant_path)

        # --- 3. 执行统计 ---
        total_count = get_database_vector_stats(qdrant_client)




# if __name__ == "__main__":
#
#     model_name = "sentence-transformers/all-mpnet-base-v2"
#     # model_name = "BAAI/bge-base-en-v1.5"
#     # model_name = "Alibaba-NLP/gte-base-en-v1.5"
#     distance_metric = "Cosine"
#     # distance_metric = "Dot"
#     # distance_metric = "Euclid"
#
#     # version_tag = datetime.now().strftime("v%Y%m%d")
#     # version_tag = "v1.0"
#     version_tag = "v1.0_7816_CPA"
#
#     model_alias = model_name.split("/")[-1]
#     base_dir = os.path.join("..", "Qdrant_Data", f"{model_alias}_{distance_metric.lower()}_{version_tag}")
#     rag_path = os.path.join("..", "RAG_Data")
#     qdrant_path = base_dir
#
#     os.makedirs(base_dir, exist_ok=True)
#
#     if not os.path.isdir(rag_path):
#         print(f"[WARN] RAG data directory not found: {rag_path}")
#
#     # Validate the distance option upfront
#     normalize_distance(distance_metric)
#
#     #使用阿里的模型时，需要用下面的，多传入一个参数
#     # embeddings = HuggingFaceEmbeddings(model_name=model_name)
#     embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"trust_remote_code": True})
#     qdrant_client = QdrantClient(path=qdrant_path)
#
#     # Inspect existing collections
#     # inspect_qdrant_collections_detailed(qdrant_client)
#
#     # Uncomment one of the following ingestion helpers when needed
#     # store_each_txt_to_db(rag_path, qdrant_client, embeddings, distance_metric)
#     # store_each_pdf_to_db(rag_path, qdrant_client, embeddings, distance_metric)
#
#     # store_single_txt_to_qdrant(os.path.join(rag_path, "some_note.txt"), qdrant_client, embeddings, "some_txt_notes", distance_metric)
#     # store_single_pdf_to_qdrant(os.path.join(rag_path, "CPA.pdf"), qdrant_client, embeddings, "CPA", distance_metric)
#
#     store_structured_json_to_qdrant(os.path.join(rag_path, "sca_rag_knowledge.json"), qdrant_client, embeddings, "long_term_memory", distance_metric)
#
#     # Delete collection examples (uncomment when needed)
#
#     # delete_qdrant_collection(qdrant_client, "long_term_memory")
#
#     # inspect_qdrant_collections_detailed(qdrant_client)
#
#     # inspect_qdrant_collection(qdrant_client=qdrant_client, collection_name="long_term_memory", preview_limit=5, show_vectors=False)
#
#
#     # Visualise vector distributions (specify collections manually)
#     # visualize_multiple_collections_vectors(qdrant_client, ["CPA", "long_term_memory"], method="umap")