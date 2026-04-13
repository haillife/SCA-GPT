# SCA-GPT

**SCA-GPT: A Generation-Planning-Tool Assisted LLM Agent for Fully Automated Side-Channel Analysis on Cryptosystems**

> Wenquan Zhou, An Wang, Yaoling Ding*, Annyu Liu, Jingqi Zhang, Jiakun Li, Liehuang Zhu
>
> *IEEE Transactions on Information Forensics and Security (TIFS)*

## Overview

Non-invasive security evaluation is essential for cryptographic devices, yet current side-channel analysis (SCA) assessments rely heavily on manual expert procedures, leading to inconsistent results, error-prone testing, and high costs. **SCA-GPT** is the first advanced LLM agent framework specifically designed for SCA tasks, enabling **fully automated ISO/IEC 17825-compliant** security evaluations through a single natural language instruction.

The framework operates through a **think-act-observe** loop and integrates three core modules:

- **Memory**: Short-term context tracking + long-term **RAG-based expert knowledge base** constructed from standards, literature, and expert entries
- **Tools**: A suite of specialized SCA tools for trace acquisition, preprocessing, TVLA, CPA, timing analysis, and more
- **Planning**: Dynamic tool selection and strategy refinement until task completion

## Key Results

| Model | Accuracy | Time Reduction |
|-------|:---:|:---:|
| Qwen3 Coder | **91.4%** | - |
| DeepSeek V3.1 | 83.8% | - |
| Kimi K2 | 77.8% | - |
| **Average** | - | **95.7%** vs. manual |

- Evaluated on **6 cryptographic algorithms**: AES, DES, RSA, ECDSA, SM3, SM4
- Deployed on **4 hardware platforms**: smart cards (7816/14443), STM32 microcontrollers, FPGA
- RAG knowledge base: **84.44%** Q@5 and **98.33%** nDCG@5
- Automated generation of ISO/IEC 17825-compliant evaluation reports

## Architecture

```
User Instruction
       |
       v
  [SCA-GPT Agent]
       |
  +----+----+----------+
  |         |          |
Memory    Planning    Tools
  |         |          |
  |    think-act-   +--+--+--+--+--+--+
  |    observe      |  |  |  |  |  |  |
  |    loop        TVLA CPA TA TC SC PP RAG
  |                 
Expert Knowledge Base
(RAG + Qdrant)
```

## Project Structure

```
tvla_agent/
├── langchain_react_agent_with_RAG_md.py  # Main agent (ReAct + RAG)
├── python_jupyter_kernel_tool.py         # Jupyter kernel backend
├── tvla_tool.py                          # TVLA analysis tool
├── timing_analysis_tool.py               # Timing analysis tool
├── trace_capture_tool.py                 # Trace capture tool
├── smartcard_tool.py                     # Smart card communication tool
├── rag_retrieve_and_summarize_tool.py    # RAG retrieval tool
├── RAG/                                  # RAG module
│   ├── Q_score.py                        # Knowledge base quality metric
│   ├── qdrant_data_generate.py           # Vector database construction
│   └── qdrant_data_search.py             # Vector database retrieval
├── RAG_Data/                             # Knowledge base source documents
│   ├── *.pdf                             # Standards & research papers
│   └── long_term_memory.json             # Expert knowledge entries
├── Qdrant_Data/                          # Pre-built vector database
│   └── all-mpnet-base-v2_cosine_v1.0/    # Embedding model: all-mpnet-base-v2
└── trace_data/                           # [Not included, see below]
```

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain langchain-openai flask jupyter_client pydantic requests
pip install matplotlib numpy scikit-learn qdrant-client sentence-transformers
```

### 2. Configure LLM API

Edit `langchain_react_agent_with_RAG_md.py` and set your API key and endpoint:

```python
openai_api_base = "your-api-base-url"
openai_api_key = "your-api-key"
```

### 3. Run

```bash
python langchain_react_agent_with_RAG_md.py
```

The agent will autonomously:
1. Retrieve relevant expert knowledge from the RAG knowledge base
2. Plan the evaluation strategy based on ISO/IEC 17825
3. Execute SCA tools (TVLA, CPA, timing analysis, etc.)
4. Verify results and iterate if needed
5. Generate an evaluation report

## Dataset

The power trace dataset used in this work covers:

| Algorithm | Platform | Traces |
|-----------|----------|--------|
| AES | Smart Card (7816) | TA + TVLA |
| AES | Smart Card (14443) | TA + TVLA |
| AES | STM32F103 | TA + TVLA (Power + EM) |
| AES | STM32F429 | TA + TVLA |
| AES | FPGA | TA + TVLA |
| DES | Smart Card (7816) | TA + TVLA |
| DES | Smart Card (14443) | TA + TVLA |
| SM3 | Smart Card (7816) | TA + TVLA |
| SM4 | Smart Card (7816) | TA + TVLA |

Due to the large size of trace files (~29GB total), the dataset is not included in this repository. To request access, please contact:

**Wenquan Zhou**: zhouwenquan@bit.edu.cn

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2025scagpt,
  title={SCA-GPT: A Generation-Planning-Tool Assisted LLM Agent for Fully Automated Side-Channel Analysis on Cryptosystems},
  author={Zhou, Wenquan and Wang, An and Ding, Yaoling and Liu, Annyu and Zhang, Jingqi and Li, Jiakun and Zhu, Liehuang},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025}
}
```

## License

This project is for academic research purposes.
