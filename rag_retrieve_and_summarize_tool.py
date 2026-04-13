from typing import Dict
from langchain.vectorstores.base import VectorStoreRetriever
from fuzzywuzzy import process, fuzz
from transformers import pipeline


def rag_retrieve_and_summarize(
    keyword: str,
    retriever_dict: Dict[str, VectorStoreRetriever],
    top_k: int = 5,
    model_name: str = "Falconsai/text_summarization"
) -> str:
    if not retriever_dict:
        return "[RAG] retriever 字典为空，无法检索。"

    file_names = list(retriever_dict.keys())
    best_match = process.extractOne(keyword, file_names, scorer=fuzz.ratio)
    if best_match is None or best_match[1] < 20:
        return f"[RAG] 未找到匹配的文档，得分过低：{best_match}"

    matched_file = best_match[0]
    retriever = retriever_dict[matched_file]
    docs = retriever.invoke(keyword)
    selected_docs = docs[:top_k]

    combined_text = "\n\n".join([doc.page_content.strip() for doc in selected_docs])
    if not combined_text.strip():
        return "[RAG] 未检索到相关内容。"

    try:
        summarizer = pipeline("summarization", model=model_name)
        # Optional truncate
        combined_text = combined_text[:3000]
        summary = summarizer(combined_text, max_length=400, min_length=50, do_sample=False)
        return f"基于关键词 \"{keyword}\" 的文档总结如下：\n\n{summary[0]['summary_text']}"
    except Exception as e:
        return f"[RAG] 执行摘要失败: {e}"
