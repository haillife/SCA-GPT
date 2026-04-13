from langchain.agents import Tool
from langchain.agents import AgentExecutor, create_react_agent
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any
import json
from langchain.vectorstores.base import VectorStoreRetriever
from rag_retrieve_and_summarize_tool import rag_retrieve_and_summarize
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from datetime import datetime
from langchain.agents import initialize_agent, AgentType
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema import Document
import os
from langchain_core.memory import BaseMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from typing import List
import socket
import subprocess
import time
from datetime import datetime
import base64
import re
import textwrap
from contextlib import redirect_stdout
import io
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from qdrant_client import QdrantClient

import re
from typing import Optional

def start_kernel_if_needed(script_path="python_jupyter_kernel_tool.py", port=5000):
    """
    检查端口，如果内核服务未运行，则在后台启动它。
    返回子进程对象，以便后续可以关闭它。
    """
    # 检查端口是否被占用
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', port)) == 0:
            print(f"⚠️ 端口 {port} 已被占用，假设内核服务已由外部启动，跳过。")
            return None  # 返回 None 表示我们没有启动它

    print(f"🚀 正在后台启动内核服务 ({script_path})...")
    # 使用 Popen 在后台启动服务，并将输出重定向，避免干扰主程序
    kernel_proc = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # 等待几秒钟，确保 Flask 服务有足够的时间完成初始化
    print("⏳ 等待内核服务初始化...")
    time.sleep(4)  # 这个等待时间很重要
    print("✅ 内核服务已启动。")

    return kernel_proc


# [MOD] 新增一个统一注入 run_dir 的 Mixin
class _RunDirMixin:
    _run_dir: str = "."

    def set_run_dir(self, run_dir: str):  # [MOD]
        self._run_dir = run_dir

    def _auto_path(self, stem: str, ext: str) -> str:  # [MOD]
        """
        在当前 run_dir 下自动生成唯一文件名（无子目录）
        例: <run_dir>/<stem>_20251108_162233_123456.<ext>
        """
        from datetime import datetime
        import os
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return os.path.join(self._run_dir, f"{stem}_{ts}.{ext.lstrip('.')}")



class JupyterNotebookInput(BaseModel):
    code: str = Field(..., description="Python code to execute in Jupyter Notebook.")


# class JupyterAPITool(BaseTool):
#     """
#     终极完美版 Jupyter 工具：
#     1. 通过代码替换，在内存中主动捕获图像为 Base64。
#     2. 通过 re.sub 和闭包，用更健壮的方式重建最终输出，确保 Markdown 链接被正确插入。
#     3. 确保生成的 Markdown 图片路径使用正斜杠 '/'，以实现最佳兼容性。
#     """
#     name: str = "JupyterAPITool"
#     description: str = "Executes Python code in a Jupyter notebook environment. It can return text output and automatically save any generated plots, embedding them in the response."
#
#     # 移除了 args_schema，让工具直接接收纯字符串输入
#     # args_schema: Type[BaseModel] = JupyterNotebookInput
#
#     image_save_path: str = "log/default_images"
#
#     def _run(self, code: str) -> str:
#         save_dir = self.image_save_path
#         os.makedirs(save_dir, exist_ok=True)
#
#         if code.startswith("```python"):
#             code = code.replace("```python", "").replace("```", "").strip()
#
#         # ==================== 新增的强力修复 ====================
#         # 在替换 plt.show() 之前，先用正则表达式注释掉所有的 plt.savefig()
#         code = re.sub(r"plt\.savefig\s*\(.*\)", "# plt.savefig() was automatically commented out by the tool.", code)
#         # =======================================================
#
#
#         replacement_code = """
# import io
# import base64
# try:
#     import matplotlib.pyplot as plt
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     plt.close()
#     buf.seek(0)
#     img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#     print(f"---IMAGE_BASE64_START---{img_b64}---IMAGE_BASE64_END---")
# except Exception as e:
#     print(f"Image capture failed: {e}")
# """
#         final_code = code.replace("plt.show()", replacement_code)
#
#         print(f"🔹 正在发送代码到 Jupyter API (主动捕获模式)...")
#
#         try:
#             response = requests.post("http://localhost:5000/execute", json={"code": final_code})
#
#             if response.status_code != 200:
#                 return f"❌ Jupyter API Error: {response.status_code} - {response.text}"
#
#             full_text_output = ""
#             outputs = response.json().get("outputs", [])
#             for item in outputs:
#                 if item.get("type") == "text":
#                     full_text_output += item.get("content", "")
#                 elif item.get("type") == "error":
#                     full_text_output += f"\n---ERROR---\n{item.get('content', '')}"
#
#             start_marker = "---IMAGE_BASE64_START---"
#             end_marker = "---IMAGE_BASE64_END---"
#             pattern = re.compile(f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", re.DOTALL)
#
#             def save_and_replace(match):
#                 img_b64 = match.group(1).strip()
#                 try:
#                     image_data = base64.b64decode(img_b64)
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#                     image_name = f"img_{timestamp}.png"
#                     image_full_path = os.path.join(save_dir, image_name)
#
#                     with open(image_full_path, "wb") as f:
#                         f.write(image_data)
#
#                     image_folder_name = os.path.basename(save_dir)
#
#                     # 核心修复：创建并返回一个总是使用正斜杠的相对路径
#                     relative_image_path = f"{image_folder_name}/{image_name}"
#
#                     print(f"🖼️ 图片已通过内存捕获并保存到 {image_full_path}")
#                     # 函数的返回值是用来替换整个匹配块的新字符串
#                     return f"![Generated Image]({relative_image_path})"
#
#                 except Exception as e:
#                     print(f"❌ 图片解码或保存失败: {e}")
#                     return f" [图片保存失败: {e}] "
#
#             # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#             # --------------------------------------------------
#
#             # 使用 re.sub 一行代码完成查找、保存、替换的全过程
#             final_observation = pattern.sub(save_and_replace, full_text_output)
#
#             print("✅ Jupyter 执行完成，已处理所有输出。")
#             return final_observation.strip()
#
#         except Exception as e:
#             error_msg = f"❌ Jupyter 工具发生未知错误: {e}"
#             print(error_msg)
#             return error_msg



# 定义输入数据

class JupyterAPITool(BaseTool):
        """
        终极稳健版 Jupyter 工具：
        1) 预处理：剥离```python外壳、统一换行、去整体缩进（dedent），根治“unexpected indent”；
        2) 注入：按 plt.show() 所在行的缩进注入图像捕获块；如用户无 show()，末尾顶格补一次；
        3) 安全：注释掉用户自写的 plt.savefig(...)，避免重复保存/冲突；
        4) 输出：从执行返回中抓取 Base64 图片，保存到 image_save_path，并替换为 Markdown 链接；
        5) 兼容：保存路径统一转为正斜杠，兼容前端渲染；其余文本/错误原样拼接返回。
        """

        name: str = "JupyterAPITool"
        description: str = (
            "Executes Python code in a Jupyter notebook environment. "
            "It returns text output and automatically saves any generated plots, embedding them as Markdown image links."
        )

        # 若需结构化入参，请取消下一行注释
        # args_schema: Type[BaseModel] = JupyterNotebookInput

        # 输出图片保存目录（相对或绝对）。建议由上层在运行前动态覆盖为 runs/<run_id>/images
        image_save_path: str = "log/default_images"

        # ===================== 内部辅助：代码预处理/注入 =====================

        @staticmethod
        def _preprocess_code(raw: str) -> str:
            """
            去掉三引号外壳、统一换行、去 BOM/不可见空格、整体去缩进。
            解决 LLM/Markdown 带来的隐形缩进与包裹问题。
            """
            s = raw.strip()
            # 去掉 ```python / ```py / ``` 头尾
            s = re.sub(r"^\s*```(?:python|py)?\s*\n?", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\n?\s*```\s*$", "", s, flags=re.IGNORECASE)
            # 统一换行 & 清除 BOM/不可见空格
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = s.lstrip("\ufeff").replace("\u00A0", " ")
            # 整体去缩进
            s = textwrap.dedent(s)
            # 去除首尾空行
            lines = s.split("\n")
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            return "\n".join(lines)

        @staticmethod
        def _indent_block(block: str, indent: str) -> str:
            """将多行 block 逐行加上指定缩进。空行保持空。"""
            return "\n".join((indent + ln if ln else ln) for ln in block.splitlines())

        @staticmethod
        def _capture_block() -> str:
            """返回用于替换 plt.show() 的无缩进图像捕获块（保持无前导空格，后续按需缩进）。"""
            return textwrap.dedent(
                """\
                import io, base64
                try:
                    import matplotlib.pyplot as plt
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    print(f"---IMAGE_BASE64_START---{img_b64}---IMAGE_BASE64_END---")
                except Exception as e:
                    print(f"Image capture failed: {e}")
                """
            ).rstrip("\n")

        # @classmethod
        # def _inject_capture_block(cls, code: str) -> str:
        #     """
        #     1) 注释掉显式 plt.savefig(...)；
        #     2) 将第一处 plt.show() 按其行缩进替换为图像捕获块；
        #     3) 若不存在 plt.show()，在末尾顶格补一个捕获块。
        #     """
        #     # 注释掉显式 savefig，避免重复保存
        #     code = re.sub(
        #         r"(?m)^\s*plt\.savefig\([^)]*\)\s*$",
        #         "# plt.savefig() was commented out by the tool.",
        #         code,
        #     )
        #
        #     pat = re.compile(r"(?m)^(?P<indent>\s*)plt\.show\(\)\s*$")
        #     block = cls._capture_block()
        #
        #     def repl(m: re.Match) -> str:
        #         indent = m.group("indent") or ""
        #         return cls._indent_block(block, indent)
        #
        #     new_code, n = pat.subn(repl, code, count=1)
        #     if n == 0:
        #         # 用户没写 show()，末尾顶格补一次
        #         new_code = code.rstrip() + "\n\n" + block + "\n"
        #     return new_code

        # =========================== 主执行入口 ============================

        # ✅ 新增：官方入口，供外部统一注入 run 目录
        def set_run_dir(self, run_dir: str):
            self.image_save_path = run_dir

        @classmethod
        def _inject_capture_block(cls, code: str) -> str:
            """
            1) 注释掉显式 plt.savefig(...)；
            2) 将第一处 plt.show() 按其行缩进替换为图像捕获块；
            3) ⬇️ [修改点] ⬇️
               若不存在 plt.show()，但代码中包含 "matplotlib"，则在末尾顶格补一个捕获块。
            """
            # 注释掉显式 savefig，避免重复保存
            code = re.sub(
                r"(?m)^\s*plt\.savefig\([^)]*\)\s*$",
                "# plt.savefig() was commented out by the tool.",
                code,
            )

            pat = re.compile(r"(?m)^(?P<indent>\s*)plt\.show\(\)\s*$")
            block = cls._capture_block()

            def repl(m: re.Match) -> str:
                indent = m.group("indent") or ""
                return cls._indent_block(block, indent)

            new_code, n = pat.subn(repl, code, count=1)

            # ⬇️ ⬇️ ⬇️ [修改点] ⬇️ ⬇️ ⬇️
            # 检查：(1) 没有替换发生 (2) 但代码中确实导入了 matplotlib
            if n == 0 and ("matplotlib" in code or "plt." in code):  # 检查更严格
                # 用户没写 show()，但导入了 plt，末尾顶格补一次
                new_code = code.rstrip() + "\n\n" + block + "\n"
            elif n == 0:
                # 没有 show() 也没有 matplotlib (例如 TraceCaptureTool)，
                # 直接返回原代码，不追加任何捕获块。
                new_code = code
            # ⬆️ ⬆️ ⬆️ [修改点] ⬆️ ⬆️ ⬆️

            return new_code
        def _run(self, code: str) -> str:


            """
            传入一段 Python 代码（可带```python外壳），在本地 Jupyter 执行，抓取文本输出和图像并返回 Markdown。
            """
            # ✅ 修改：优先使用环境变量，其次用实例字段，最后兜底默认
            save_dir = (
                    os.getenv("AGENT_RUN_DIR")
                    or getattr(self, "image_save_path", None)
                    or "log/default_images"
            )
            os.makedirs(save_dir, exist_ok=True)

            print(f"🗂️ Jupyter image_save_path = {save_dir}")

            # 1) 预处理：剥壳/换行/BOM/不可见空格/整体去缩进
            code = self._preprocess_code(code)

            # 2) 注入：按 plt.show() 行缩进替换图像捕获块；无 show() 则末尾补一个
            code = self._inject_capture_block(code)

            print("🔹 正在发送代码到 Jupyter API (主动捕获模式)...")

            try:
                # 3) 调用你的 Jupyter 执行服务
                resp = requests.post("http://localhost:5000/execute", json={"code": code})
                if resp.status_code != 200:
                    return f"❌ Jupyter API Error: {resp.status_code} - {resp.text}"

                # 4) 汇总文本输出 & 错误
                full_text_output = ""
                outputs = resp.json().get("outputs", [])
                for item in outputs:
                    if item.get("type") == "text":
                        full_text_output += item.get("content", "")
                    elif item.get("type") == "error":
                        full_text_output += f"\n---ERROR---\n{item.get('content', '')}"

                # 5) 处理 Base64 图片 → 文件 → Markdown
                start_marker = "---IMAGE_BASE64_START---"
                end_marker = "---IMAGE_BASE64_END---"
                pattern = re.compile(
                    f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}",
                    re.DOTALL,
                )

                def save_and_replace(match: re.Match) -> str:
                    img_b64 = match.group(1).strip()
                    try:
                        image_data = base64.b64decode(img_b64)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        image_name = f"img_{ts}.png"
                        image_full_path = os.path.join(save_dir, image_name)
                        with open(image_full_path, "wb") as f:
                            f.write(image_data)

                        # 统一正斜杠，前端 Markdown 更稳
                        # rel = f"{os.path.basename(save_dir)}/{image_name}".replace("\\", "/")
                        rel = image_name
                        print(f"🖼️ 图片已通过内存捕获并保存到 {image_full_path}")
                        return f"![Generated Image]({rel})"
                    except Exception as e:
                        print(f"❌ 图片解码或保存失败: {e}")
                        return f"[图片保存失败: {e}]"

                final_observation = pattern.sub(save_and_replace, full_text_output)

                print("✅ Jupyter 执行完成，已处理所有输出。")
                return final_observation.strip()

            except Exception as e:
                err = f"❌ Jupyter 工具发生未知错误: {e}"
                print(err)
                return err


class TAInput(BaseModel):
    json: str = Field(..., description="A JSON string containing npy_file, num_traces, num_points, sampling_rate and clock_freq.")


# 定义 Timing Analysis 工具类
class TATool(BaseTool):
    name: str = "TimingAnalyzer"
    description: str = (
        "执行 Timing Analysis（TA）测试，返回固定输入与随机输入平均穿越阈值时间之差 |T1 - T2|（单位：秒）。\n"
        "输入格式为 JSON 字符串，必须包含 npy_file, num_traces, num_points, sampling_rate，"
        "可选 voltage_threshold（未提供则自动计算）。"
    )
    args_schema: Type[BaseModel] = TAInput

    def _run(self, json_input: str) -> str:
        import json
        try:
            data = json.loads(json_input)
            npy_file = data["npy_file"]
            num_traces = data["num_traces"]
            num_points = data["num_points"]
            sampling_rate = data["sampling_rate"]
            clock_freq = data["clock_freq"]
            voltage_threshold = data.get("voltage_threshold", None)
        except Exception as e:
            return f"[TimingAnalyzer] Parameter resolution failed：{e}"

        code = f"""
from timing_analysis_tool import compute_timing_difference

delta = compute_timing_difference(
    npy_file="{npy_file}",
    num_traces={num_traces},
    num_points={num_points},
    sampling_rate={sampling_rate},
    clock_freq={clock_freq},
    {"voltage_threshold=" + str(voltage_threshold) + "," if voltage_threshold is not None else ""}
)

"""

        return JupyterAPITool().run(code)


class TVLAInput(BaseModel):
    # 使用清晰的名称 json_input，不再使用别名
    json_input: str = Field(..., description="一个包含 npy_file, num_traces, num_points 的 JSON 字符串")

class TVLATool(BaseTool):
    name: str = "TVLAAnalyzer"
    description: str = "输入参数为：npy 文件、波形条数、点数。根据输入的参数，生成一段用于执行 TVLA 分析和绘图的 Python 代码。你应该在下一步中使用 JupyterAPITool 来执行这段返回的代码。"
    args_schema: Type[BaseModel] = TVLAInput

    # _run 方法的参数名与 Pydantic 模型中的字段名完全一致
    def _run(self, json_input: str) -> str:
        try:
            data = json.loads(json_input)
            # ... (后续代码不变)
            npy_file = data["npy_file"]
            num_traces = data["num_traces"]
            num_points = data["num_points"]
        except Exception as e:
            # 如果参数解析失败，返回一段注释代码，告知错误
            return f"# [TVLAAnalyzer] 参数解析失败：{e}"

        code = f"""
from tvla_tool import compute_tvla_general
import matplotlib.pyplot as plt
t_values = compute_tvla_general("{npy_file}", {num_traces}, {num_points})
# Add plotting
plt.figure(figsize=(12, 6))
plt.plot(t_values, 'b-', label='TVLA Result')
plt.axhline(y=4.5, color='r', linestyle='--', label='Threshold (4.5)')
plt.xlabel('Point Index')
plt.ylabel('t-value')
plt.title('TVLA Analysis Result')
plt.legend()
plt.show()

# 打印结果
print("t_values[:10]:", t_values[:10])
print("Max:", max(t_values))
print("Min:", min(t_values))
"""
        return code

#
# - If you use the SPAAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_segments": 36, "use_mean_over_traces": true}}. This tool only returns reference Python code for SPA; you may execute it as-is with the JupyterAPITool, modify the code and then execute it, or simply choose not to use that code.
# class SPAInput(BaseModel):
#     # 使用清晰的名称 json_input，内容至少包含：npy_file, num_segments
#     json_input: str = Field(
#         ...,
#         description="一个包含 npy_file, num_segments 的 JSON 字符串，可选 use_mean_over_traces(默认True)。"
#     )
#
# class SPATool(BaseTool):
#     name: str = "SPAAnalyzer"
#     description: str = (
#         "输入参数为：npy 文件路径 和 段数。将波形按段数近似等分（不可整除时自动均摊），"
#         "每段插值到统一长度，做标准化后 PCA 降到 2 维，再用 KMeans 聚 2 类，并打印每段的聚类标签；"
#         "This tool only returns reference Python code for SPA; you may execute it as-is with the JupyterAPITool, modify the code and then execute it, or simply choose not to use that code."
#     )
#     args_schema: Type[BaseModel] = SPAInput
#
#     def _run(self, json_input: str) -> str:
#         try:
#             data = json.loads(json_input)
#             npy_file = data["npy_file"]
#             num_segments = int(data["num_segments"])
#             use_mean_over_traces = bool(data.get("use_mean_over_traces", True))
#         except Exception as e:
#             return f"# [SPAAnalyzer] 参数解析失败：{e}"
#
#         code = f"""
# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
#
# # ------------- 配置 -------------
# npy_file = r"{npy_file}"
# num_segments = int({num_segments})
# use_mean_over_traces = {str(use_mean_over_traces)}
#
# # ------------- 工具函数 -------------
# def _resample_to_length(arr, target_len):
#     \"\"\"将一维数组 arr 线性插值到 target_len 长度（仅依赖 numpy）。\"\"\"
#     if len(arr) == target_len:
#         return arr.astype(float)
#     if len(arr) < 2:
#         # 退化情况：长度<2，直接重复到目标长度
#         return np.full(target_len, float(arr[0] if len(arr) == 1 else 0.0))
#     x_old = np.linspace(0.0, 1.0, num=len(arr))
#     x_new = np.linspace(0.0, 1.0, num=target_len)
#     return np.interp(x_new, x_old, arr).astype(float)
#
# def _approx_equal_split_indices(total_len, n):
#     \"\"\"把 [0, total_len) 均匀切成 n 段，余数从前往后均摊，返回每段的 (start, end)。\"\"\"
#     base = total_len // n
#     rem = total_len % n
#     idx = 0
#     spans = []
#     for i in range(n):
#         extra = 1 if i < rem else 0
#         start = idx
#         end = start + base + extra
#         spans.append((start, end))
#         idx = end
#     return spans
#
# # ------------- 加载与预处理 -------------
# if not os.path.exists(npy_file):
#     raise FileNotFoundError(f"npy 文件不存在: {{npy_file}}")
#
# arr = np.load(npy_file, allow_pickle=False)
#
# # 允许 arr 为 (N,) 或 (T, N)。若为 (T, N)，可选择对 T 条波形做均值合成
# if arr.ndim == 1:
#     signal = arr.astype(float)
# elif arr.ndim == 2:
#     if use_mean_over_traces:
#         signal = arr.mean(axis=0).astype(float)
#     else:
#         # 不做均值时，默认取首条
#         signal = arr[0].astype(float)
# else:
#     raise ValueError(f"不支持的 npy 形状: {{arr.shape}}，期望 1D 或 2D")
#
# total_points = len(signal)
# if num_segments <= 1 or num_segments > total_points:
#     raise ValueError(f"num_segments={{num_segments}} 非法，应在 [2, {{total_points}}] 范围内")
#
# # ------------- 近似等分 -------------
# spans = _approx_equal_split_indices(total_points, num_segments)
# lengths = [end - start for start, end in spans]
# target_len = int(np.median(lengths))  # 统一插值到“中位长度”，更稳
# target_len = max(target_len, 2)
#
# segments = []
# for i, (start, end) in enumerate(spans):
#     seg = signal[start:end]
#     seg_r = _resample_to_length(seg, target_len)  # 插值到统一长度
#     segments.append(seg_r)
#
# X = np.vstack(segments)  # 形状: (num_segments, target_len)
#
# # ------------- 标准化 + PCA(2D) -------------
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# pca = PCA(n_components=2, random_state=42)
# X_2d = pca.fit_transform(X_std)
#
# # ------------- KMeans 聚类（2 类） -------------
# kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
# labels = kmeans.fit_predict(X_2d)
#
# # ------------- 输出与可视化 -------------
# print("总点数:", total_points)
# print("段数:", num_segments)
# print("每段统一长度(插值后):", target_len)
# print("PCA 方差解释率:", pca.explained_variance_ratio_)
# print("KMeans 簇中心(2D):\\n", kmeans.cluster_centers_)
# print("每段的聚类标签(按段索引顺序):")
# print(labels.tolist())
#
# # 统计每类包含的段索引
# cluster0_idx = [i for i, lb in enumerate(labels) if lb == 0]
# cluster1_idx = [i for i, lb in enumerate(labels) if lb == 1]
# print(f"Cluster 0 段索引: {{cluster0_idx}} (共 {{len(cluster0_idx)}})")
# print(f"Cluster 1 段索引: {{cluster1_idx}} (共 {{len(cluster1_idx)}})")
#
# # 绘制 PCA 2D 散点并标注段号
# plt.figure(figsize=(8, 6))
# plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=60)
# for i, (x, y) in enumerate(X_2d):
#     plt.text(x, y, str(i), fontsize=9, ha='center', va='center')
#
# plt.title("SPA: PCA(2D) + KMeans(2类)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()
# """
#         return code


# 定义输入数据
class TraceCaptureInput(BaseModel):
    json: str = Field(..., description="一个包含 npy_file, point_num, trace_num 的 JSON 字符串")

# 定义TraceCapture工具，继承 BaseTool
# 帮我采集1条智能卡上AES加密的波形，24万点。
# - If you use the TraceCapture tool, your Action Input must provide a single-line JSON input like "json": {{\"npy_file\": \"path/to/file.npy\", \"point_num\": 240000, \"trace_num\": 1}}.
class TraceCaptureTool(BaseTool):
    name: str = "TraceCapture"
    description: str = (
        "Capture traces, save to file, and plot the first trace as a preview."  # <--- [修改] 描述更新
    )
    args_schema: Type[BaseModel] = TraceCaptureInput
    jupyter_tool: JupyterAPITool  # <--- 保持不变

    def _run(self, json_input: str) -> str:
        try:
            data = json.loads(json_input)
            npy_file = data["npy_file"]
            point_num = data["point_num"]
            trace_num = data["trace_num"]
        except Exception as e:
            return f"[TraceCapture] 参数解析失败：{e}"

        # [修改] code 字符串现在包含了绘图逻辑
        code = f"""
import numpy as np
import matplotlib.pyplot as plt
from trace_capture_tool import trace_capture_nt

# 定义参数
npy_file = "{npy_file}"
point_num = {point_num}
trace_num = {trace_num}

# 执行采集
trace_capture_nt(npy_file, point_num=point_num, trace_num=trace_num)

print("Trace captured and saved to:", npy_file)

# ⬇️ ⬇️ ⬇️ [新增的绘图部分] ⬇️ ⬇️ ⬇️
# 作为预览，加载刚保存的数据并绘制第一条波形
if trace_num > 0:
    try:
        # 加载刚刚保存的 .npy 文件
        traces = np.load(npy_file)

        # 假设波形形状为 (points, traces)，例如 (600000, 1)
        # 我们只绘制第一条 (index 0)
        if traces.ndim == 2:
            first_trace = traces[:, 0]
        elif traces.ndim == 1: # 兼容只采一条时可能保存为1D的情况
            first_trace = traces
        else:
            raise Exception(f"无法解析波形形状: {{traces.shape}}")

        plt.figure(figsize=(12, 6))
        plt.plot(first_trace, 'b-', label='First Captured Trace (Trace 0)')
        plt.title(f'Trace Capture Preview (Displaying 1 of {trace_num} traces)')
        plt.xlabel('Sample Point Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        # 关键: plt.show() 会被 JupyterAPITool 捕获
        plt.show()

    except Exception as e:
        print(f"--- PLOTTING FAILED ---")
        print(f"波形图绘制失败: {{e}}")
        print(f"注意: 波形采集和保存已成功完成于 {npy_file}，仅预览失败。")
# ⬆️ ⬆️ ⬆️ [新增的绘图部分] ⬆️ ⬆️ ⬆️
"""

        clean_code = textwrap.dedent(code)
        # 保持不变：仍然使用 JupyterAPITool 来执行
        return self.jupyter_tool.run(clean_code)


# ---------------- 1) 低通滤波封装 ----------------
class LowpassInput(BaseModel):
    # [MOD] 只保留必要参数；删除 out_wave_npy、plot_path
    json: str = Field(
        ...,
        description=(
            "JSON with: wave_npy (str), weight (int, default 10), trace_index (int, default 0)."
        ),
    )

class LowpassTool(_RunDirMixin, BaseTool):  # [MOD] 继承 _RunDirMixin
    name: str = "Lowpass_Filter"
    description: str = (
        "Weighted low-pass IIR filter s_i=(w*s_{i-1}+x_i)/(1+w). "
        "Generates filtered waves and a comparison plot."
    )
    args_schema: Type[BaseModel] = LowpassInput

    def _run(self, json_input: str) -> str:
        try:
            data = json.loads(json_input)
            wave_npy = data["wave_npy"]
            weight = int(data.get("weight", 10))
            trace_index = int(data.get("trace_index", 0))
        except Exception as e:
            return f"[Lowpass_Filter] Parameter resolution failed: {e}"

        # [MOD] 统一自动命名到 run_dir
        out_wave_npy = self._auto_path("lowpass", "npy")           # [MOD]
        plot_path    = self._auto_path("lowpass_compare", "png")   # [MOD]

        code = f"""
import numpy as np, json as _json
import matplotlib.pyplot as plt
from preprocessing_tool import lowpass_iir_weighted

waves = np.load({json.dumps(wave_npy)})
n_traces, n_points = waves.shape
idx = max(0, min({trace_index}, n_traces-1))

filt = lowpass_iir_weighted(waves, w={weight})
np.save({json.dumps(out_wave_npy)}, filt.astype(waves.dtype, copy=False))

plt.figure(figsize=(10,3))
plt.plot(waves[idx], linewidth=0.9, label=f"raw[{{idx}}]")
plt.plot(filt[idx],  linewidth=1.2, label=f"filtered[{{idx}}]")
plt.title(f"Low-pass compare (w={weight})")
plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.legend()
plt.tight_layout(); plt.savefig({json.dumps(plot_path)}, dpi=160); plt.close()

print(_json.dumps({{
  "in_shape": tuple(waves.shape),
  "out_shape": tuple(filt.shape),
  "weight": {weight},
  "trace_index": int(idx),
  "out_wave_npy": {json.dumps(out_wave_npy)},
  "plot_path": {json.dumps(plot_path)}
}}, ensure_ascii=False))
"""
        return JupyterAPITool().run(code)

# ---------------- 2) 锚点选择封装 ----------------
class AnchorPickInput(BaseModel):
    json: str = Field(
        ...,
        description=(
            "A JSON string with fields: wave_npy, "
            "weight (int, default 10), min_distance (int, default 300), edge_margin (int, default 300), "
            "default_win_ratio (float, default 0.025), max_offset_ratio (float, default 0.5), threshold (int, default 95)."
        ),
    )

class AnchorPickTool(BaseTool):
    name: str = "Anchor_Picker"
    description: str = (
        "Automatically select anchor position and output static-alignment params as a string "
        "{ref_trace,start,length,max_shift,threshold} without saving any file."
    )
    args_schema: Type[BaseModel] = AnchorPickInput

    def _run(self, json_input: str) -> str:
        try:
            data = json.loads(json_input)
            wave_npy = data["wave_npy"]
            weight = int(data.get("weight", 10))
            min_distance = int(data.get("min_distance", 100))
            edge_margin = int(data.get("edge_margin", 0))
            default_win_ratio = float(data.get("default_win_ratio", 0.015))
            max_offset_ratio = float(data.get("max_offset_ratio", 0.05))
            threshold = int(data.get("threshold", 90))
        except Exception as e:
            return f"[Anchor_Picker] Parameter resolution failed: {e}"

        code = f"""
import numpy as np, json as _json
from preprocessing_tool import pick_anchor_params

waves = np.load({json.dumps(wave_npy)})
params = pick_anchor_params(
    waves,
    weight={weight},
    min_distance={min_distance},
    edge_margin={edge_margin},
    default_win_ratio={default_win_ratio},
    max_offset_ratio={max_offset_ratio},
    threshold={threshold}
)

# 打印输出参数字符串
out_str = _json.dumps(params, ensure_ascii=False)
print(out_str)
"""
        return JupyterAPITool().run(code)

# ---------------- 3) 静态对齐封装（保存对齐结果；叠加3条波展示） ----------------
class StaticAlignInput(BaseModel):
    # [MOD] 删除 out_aligned、plot_path；保留业务参数
    json: str = Field(
        ...,
        description=(
            "JSON with: wave_npy (str), ref_trace (int), start (int), length (int), "
            "max_shift (int), threshold (int 1..100), show_indices (optional list[int])."
        ),
    )

class StaticAlignTool(_RunDirMixin, BaseTool):  # [MOD]
    name: str = "Static_Aligner"
    description: str = (
        "Static alignment using provided scalar params. "
        "Saves the aligned waveform array and overlays traces."
    )
    args_schema: Type[BaseModel] = StaticAlignInput

    def _run(self, json_input: str) -> str:
        try:
            data = json.loads(json_input)
            wave_npy    = data["wave_npy"]
            ref_trace   = int(data["ref_trace"])
            start       = int(data["start"])
            length      = int(data["length"])
            max_shift   = int(data["max_shift"])
            threshold   = int(data["threshold"])
            show_indices= data.get("show_indices", None)
        except Exception as e:
            return f"[Static_Aligner] Parameter resolution failed: {e}"

        # [MOD] 统一自动命名到 run_dir
        out_aligned = self._auto_path("aligned", "npy")            # [MOD]
        plot_path   = self._auto_path("align_overlay", "png")      # [MOD]
        jb_indices = json.dumps(show_indices) if isinstance(show_indices, list) else "None"

        code = f"""
import numpy as np, json as _json
import matplotlib.pyplot as plt
from preprocessing_tool import static_align

waves = np.load({json.dumps(wave_npy)})

params = {{
    "ref_trace": {ref_trace},
    "start": {start},
    "length": {length},
    "max_shift": {max_shift},
    "threshold": {threshold}
}}

ret = static_align(waves, params)
aligned, shifts, accepted = ret['aligned'], ret['shifts'], ret['accepted']

np.save({json.dumps(out_aligned)}, aligned)

idxs = {jb_indices}
if idxs is None:
    ok = np.where(accepted)[0]
    if len(ok) >= 3:
        idxs = ok[:3].tolist()
    else:
        idxs = list(range(min(3, aligned.shape[0])))

plt.figure(figsize=(10,3))
for idx in idxs:
    if 0 <= idx < aligned.shape[0]:
        plt.plot(aligned[idx], linewidth=1.0, label=f"t{{idx}}")
plt.title("Static alignment overlay")
plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.legend()
plt.tight_layout(); plt.savefig({json.dumps(plot_path)}, dpi=160); plt.close()

print(_json.dumps({{
  "in_shape": tuple(waves.shape),
  "aligned_shape": tuple(aligned.shape),
  "accepted": int(accepted.sum()),
  "rejected": int((~accepted).sum()),
  "indices_shown": idxs,
  "out_aligned": {json.dumps(out_aligned)},
  "plot_path": {json.dumps(plot_path)}
}}, ensure_ascii=False))
"""
        return JupyterAPITool().run(code)


class CPAInput(BaseModel):
    # [MOD] 删除 plot_path；保留业务参数
    json: str = Field(
        ...,
        description=(
            "JSON with: plaintext_npy (str), wave_npy (str), candidates (int, default 4), "
            "leakage_model ('HW'|'ID', default 'HW'), leakage_position ('S-box_out'), "
            "bytes_to_test (optional list[int])."
        ),
    )

class CPATool(_RunDirMixin, BaseTool):  # [MOD]
    name: str = "AES_CPA_Analyzer"
    description: str = (
        "Perform AES-128 CPA against first-round intermediate. "
        "Returns per-byte top-K key candidates with correlation scores and time indices."
    )
    args_schema: Type[BaseModel] = CPAInput

    def _run(self, json_input: str) -> str:
        try:
            data = json.loads(json_input)
            plaintext_npy   = data["plaintext_npy"]
            wave_npy        = data["wave_npy"]
            candidates      = int(data.get("candidates", 4))
            leakage_model   = data.get("leakage_model", "HW")
            leakage_position= data.get("leakage_position", "S-box_out")
            bytes_to_test   = data.get("bytes_to_test", None)
        except Exception as e:
            return f"[AES_CPA_Analyzer] Parameter resolution failed: {e}"

        # [MOD] 统一自动命名到 run_dir
        plot_path = self._auto_path("cpa_corr", "png")  # [MOD]
        jb = json.dumps(bytes_to_test) if bytes_to_test is not None else "None"

        code = f"""
from aes_cpa_tool import run_aes_cpa
import json as _json

res = run_aes_cpa(
    plaintext_npy={json.dumps(plaintext_npy)},
    wave_npy={json.dumps(wave_npy)},
    candidates={candidates},
    leakage_model={json.dumps(leakage_model)},
    leakage_position={json.dumps(leakage_position)},
    bytes_to_test={jb},
    plot_path={json.dumps(plot_path)}
)
print(_json.dumps(res, ensure_ascii=False))
print(res.get("plot_path", ""))
"""
        return JupyterAPITool().run(code)


# # 定义输入数据
# class RAG_retrieve_and_summarize_Input(BaseModel):
#     json: str = Field(..., description="一个包含 keyword 的 JSON 字符串，形如 {\"keyword\": \"功耗攻击\"}")
#
# # 定义RAG_retrieve_and_summarize工具，继承 BaseTool
# class RAG_retrieve_and_summarize_Tool(BaseTool):
#     name: str = "RAG_retrieve_and_summarize"
#     description: str = (
#         "输入一个关键词，模糊匹配向量数据库中最相关的文档内容，并对前5个片段做摘要总结。"
#     )
#     args_schema: Type[BaseModel] = RAG_retrieve_and_summarize_Input
#
#     retriever_dict: Dict[str, VectorStoreRetriever]
#
#     def _run(self, json_input: str) -> str:
#         try:
#             data = json.loads(json_input)
#             keyword = data["keyword"]
#         except Exception as e:
#             return f"[TraceCapture] 参数解析失败：{e}"
#
#         return rag_retrieve_and_summarize(keyword, self.retriever_dict)



# # 遍历 collection 构造 retriever_dict，后面用RAG_retrieve_and_summarize工具就是要把这个传进去，才能进行搜索匹配
# retriever_dict = {
#     name: Qdrant(
#         client=qdrant_client,
#         collection_name=name,
#         embeddings=embeddings
#     ).as_retriever()
#     for name in [c.name for c in qdrant_client.get_collections().collections]
# }

# 进行数据库准备，数据库主要分为两部分，专业知识材料以及优秀的长期记忆
# retriever_dict 是用于构建 Tool（RAG查阅知识库），多 Collection，对应不同文档
# long_term_memory 是 Agent 的内置记忆系统，仅一个 Collection，自动注入 Prompt

# ==== 启动 Qdrant Client ====



# ==== 短期记忆 ==== 也就是上下文情景记忆context  ==没用，这里不涉及到多轮对话，所以实际没内容
# short_term_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ==== 长期记忆 ====  也就是多轮对话、多智能体、多、、各种记忆都向量化的存储在数据库中，可以进行任意搜索匹配，但目前的方案是仅在第一次query的时候进行匹配，中间Thought、Observation不进行自动匹配，如果需要可以用mem0
# VectorStoreRetrieverMemory默认会把当前的记录写入数据库，要关掉
# 默认的数据导入函数也不行，会把metadata吃掉，所以重新写了一个memory类，用来实现对长期记忆的检索
# 这里只是搜索了一个字典，改为全部
# class FixedQdrantRetriever:
#     def __init__(self, qdrant_client, collection_name, embeddings, top_k=5, threshold=0.1):
#         self.qdrant_client = qdrant_client
#         self.collection_name = collection_name
#         self.embeddings = embeddings
#         self.k = top_k
#         self.threshold = threshold
#
#     def get_relevant_documents(self, query: str):
#         query_vector = self.embeddings.embed_query(query)
#         result = self.qdrant_client.search(
#             collection_name=self.collection_name,
#             query_vector=query_vector,
#             limit=self.k,
#             score_threshold=self.threshold,
#             with_payload=True
#         )
#         docs = []
#         for pt in result:
#             metadata = pt.payload or {}
#             docs.append(Document(
#                 page_content=metadata.get("text", "[无内容]"),
#                 metadata=metadata
#             ))
#         return docs
#
# class FixedRetrieverMemory(BaseMemory):
#     retriever: Any = Field()  # ✅ Pydantic 字段声明
#     memory_key: str = "history"
#
#     def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
#         query = inputs.get("input", "")
#         docs = self.retriever.get_relevant_documents(query)
#         combined = "\n\n".join(f"【{doc.metadata.get('title', '无标题')}】\n{doc.page_content}" for doc in docs)
#         return {self.memory_key: combined}
#
#     def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
#         pass  # 禁止写入
#
#     def clear(self) -> None:
#         pass
#
#     @property
#     def memory_variables(self) -> List[str]:
#         return [self.memory_key]
#
# memory_collection_name = "long_term_memory"
# retriever = FixedQdrantRetriever(
#     qdrant_client=qdrant_client,
#     collection_name=memory_collection_name,
#     embeddings=embeddings,
#     top_k=5,        #这里代表要搜索到几条数据
#     threshold=0.1   #这里代表搜索的数据最低跟query的相关性
# )
#
# long_term_memory = FixedRetrieverMemory(retriever=retriever)



#
#
# class MultiCollectionQdrantRetriever:
#     def __init__(self, qdrant_client, collection_names, embeddings, top_k=5, threshold=0.1):
#         self.qdrant_client = qdrant_client
#         self.collection_names = collection_names if isinstance(collection_names, list) else [collection_names]
#         self.embeddings = embeddings
#         self.k = top_k
#         self.threshold = threshold
#
#     def get_relevant_documents(self, query: str) -> List[Document]:
#         query_vector = self.embeddings.embed_query(query)
#
#         # 收集所有集合的候选（每个集合各取Top-K，随后全局排序）
#         candidates = []  # (score, Document)
#
#         for collection in self.collection_names:
#             result = self.qdrant_client.search(
#                 collection_name=collection,
#                 query_vector=query_vector,
#                 limit=self.k,                 # 每个集合各取K
#                 score_threshold=self.threshold,
#                 with_payload=True
#             )
#             for pt in result:
#                 payload = pt.payload or {}
#                 # 构造 Document，带上集合名、point id 和原始score
#                 meta = dict(payload)
#                 meta.setdefault("title", payload.get("title", "无标题"))
#                 meta["collection"] = collection
#                 meta["qdrant_id"] = getattr(pt, "id", None)
#                 meta["score"] = getattr(pt, "score", None)
#
#                 doc = Document(
#                     page_content=payload.get("text", "[无内容]"),
#                     metadata=meta
#                 )
#                 candidates.append((pt.score, doc))
#
#         # 全局按分数降序，取前K
#         candidates.sort(key=lambda x: (x[0] if x[0] is not None else float("-inf")), reverse=True)
#         top_docs = [doc for _, doc in candidates[:self.k]]
#         return top_docs
#
# class MultiCollectionQdrantRetrieverMemory(BaseMemory):
#     retriever: Any = Field()  # ✅ Pydantic 字段声明
#     memory_key: str = "history"
#
#     def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
#         query = inputs.get("input", "")
#         docs = self.retriever.get_relevant_documents(query)
#         combined = "\n\n".join(f"【{doc.metadata.get('title', '无标题')}】\n{doc.page_content}" for doc in docs)
#         return {self.memory_key: combined}
#
#     def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
#         pass  # 禁止写入
#
#     def clear(self) -> None:
#         pass
#
#     @property
#     def memory_variables(self) -> List[str]:
#         return [self.memory_key]
#
#
#
# qdrant_path = "./Qdrant_Data/all-mpnet-base-v2_cosine_v1.0"
# qdrant_client = QdrantClient(path=qdrant_path)
#
# # ==== 使用 HF 的 embedding 模型 维度768,这里模型要跟数据库匹配====
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#
# collections = qdrant_client.get_collections()
# all_collections = [c.name for c in collections.collections]
#
# multi_retriever = MultiCollectionQdrantRetriever(
#     qdrant_client=qdrant_client,
#     collection_names=all_collections,
#     embeddings=embeddings,
#     top_k=5,  # 这里代表要搜索到几条数据
#     threshold=0.1  # 这里代表搜索的数据最低跟query的相关性
# )
#
# long_term_memory = MultiCollectionQdrantRetrieverMemory(retriever=multi_retriever)

# class HybridMemory(BaseMemory):
#     """合并短期和长期记忆为一个统一接口"""
#     def __init__(self, short_memory, long_memory):
#         self.short = short_memory
#         self.long = long_memory
#
#     def load_memory_variables(self,  inputs: Dict[str, Any]) -> Dict[str, Any]:
#         short_vars = self.short.load_memory_variables(inputs)   # {"chat_history": ...}
#         long_vars = self.long.load_memory_variables(inputs)     # {"history": ...}
#
#         return {
#             "chat_history": short_vars.get("chat_history", ""),
#             "long_term_memory": long_vars.get("history", "")  # 注意：这里用 "history"
#         }
#
#     def save_context(self, inputs, outputs):
#         self.short.save_context(inputs, outputs)
#         self.long.save_context(inputs, outputs)
#
#     def clear(self):
#         self.short.clear()
#         self.long.clear()
#
# # 初始化
# hybrid_memory = HybridMemory(short_memory=short_term_memory, long_memory=long_term_memory)


def _cosine(a, b) -> float:
    va = np.array(a); vb = np.array(b)
    na = np.linalg.norm(va); nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))

def _normalize_score(raw_score: float, distance_mode: str) -> float:
    """
    统一成“越大越好”的相似度：
      - cosine/dot: 直接返回 raw_score
      - euclid:     相似度 = 1/(1+d)
    """
    dm = (distance_mode or "cosine").lower()
    if dm in ("euclid", "euclidean", "l2"):
        d = max(float(raw_score), 0.0)
        return 1.0 / (1.0 + d)
    return float(raw_score)


def infer_distance_mode_from_path(db_path: str, default: str = "cosine") -> str:
    """
    从数据库路径名里推断距离度量（大小写不敏感）。
    支持: "cosine" | "dot" | "euclid"/"euclidean"/"l2"
    """
    s = (db_path or "").lower()

    # 先匹配最明确的关键词
    if re.search(r"(?:^|[_\-./])cosine(?:$|[_\-./])", s):
        return "cosine"
    if re.search(r"(?:^|[_\-./])dot(?:$|[_\-./])", s):
        return "dot"
    if re.search(r"(?:^|[_\-./])(euclid|euclidean|l2)(?:$|[_\-./])", s):
        return "euclid"

    # 没有命中就回退默认
    return default


class MultiCollectionQdrantRetrieverLTM:
    """
    仅对 long_term_collection 应用：标题相似度权重 + 集合权重。
    其他 collection 仅用文本的 base 相似度（Qdrant score 归一后）。
    """
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_names: List[str],
        embeddings,
        *,
        top_k: int = 5,
        score_threshold: Optional[float] = None,     # euclid 下表示“最大距离”，不确定可 None
        per_collection_fetch: Optional[int] = None,  # 每库抓取多少候选再合并排序
        long_term_collection: str = "long_term_memory",
        collection_boost: float = 0.0,               # 仅对 long_term_collection 生效，如 0.35
        title_sim_weight: float = 0.30,              # 仅对 long_term_collection 生效
        combine_mode: str = "mul",                   # "mul"（稳）或 "add"（激进）
        distance_mode: str = "cosine",               # 全局距离："cosine"|"dot"|"euclid"
    ):
        self.client = qdrant_client
        self.collections = collection_names if isinstance(collection_names, list) else [collection_names]
        self.embeddings = embeddings
        self.k = top_k
        self.score_threshold = score_threshold
        self.per_collection_fetch = per_collection_fetch
        self.long_term_collection = long_term_collection
        self.collection_boost = float(collection_boost)
        self.title_sim_weight = float(title_sim_weight)
        self.combine_mode = combine_mode
        self.distance_mode = distance_mode

        # 标题向量缓存，避免重复计算
        self._title_vec_cache: Dict[str, List[float]] = {}

    def _final_score(
        self,
        *,
        base_sim: float,
        is_ltm: bool,
        title_sim: Optional[float] = None
    ) -> float:
        """
        乘法融合（默认）：
            final = base_sim * (1 + collection_boost) * (1 + title_sim_weight * title_sim)
        加法融合：
            final = base_sim + collection_boost + title_sim_weight * title_sim
        仅 is_ltm=True 时才启用 collection_boost 与 title_sim。
        """
        if not is_ltm:
            return base_sim

        t = float(title_sim or 0.0)
        if self.combine_mode == "add":
            return base_sim + self.collection_boost + self.title_sim_weight * t
        else:  # "mul"
            return base_sim * (1.0 + self.collection_boost) * (1.0 + self.title_sim_weight * t)

    def get_relevant_documents(self, query: str) -> List[Document]:
        qvec = self.embeddings.embed_query(query)
        per_k = self.per_collection_fetch if self.per_collection_fetch else max(self.k, 10)

        candidates: List[Tuple[float, Document]] = []

        for col in self.collections:
            kwargs = dict(
                collection_name=col,
                query_vector=qvec,
                limit=per_k,
                with_payload=True,
            )
            if self.score_threshold is not None:
                kwargs["score_threshold"] = self.score_threshold

            results = self.client.search(**kwargs)
            is_ltm = (col == self.long_term_collection)

            for pt in results:
                payload: Dict[str, Any] = pt.payload or {}
                title = payload.get("title", "无标题")
                text = payload.get("text", "[无内容]")
                pid = getattr(pt, "id", None)
                raw_score = float(getattr(pt, "score", 0.0))

                # 1) 归一基础分
                base_sim = _normalize_score(raw_score, self.distance_mode)

                # 2) 仅 LTM 计算标题相似度并融合
                title_sim = None
                if is_ltm:
                    if title in self._title_vec_cache:
                        tvec = self._title_vec_cache[title]
                    else:
                        tvec = self.embeddings.embed_query(title)
                        self._title_vec_cache[title] = tvec
                    title_sim = _cosine(qvec, tvec)

                final = self._final_score(base_sim=base_sim, is_ltm=is_ltm, title_sim=title_sim)

                # 3) 构造 Document（保留调参信息，便于 debug）
                meta = dict(payload)
                meta.setdefault("title", title)
                meta["collection"] = col
                meta["qdrant_id"] = pid
                meta["raw_score"] = raw_score
                meta["base_sim"] = base_sim
                if is_ltm:
                    meta["title_sim"] = title_sim
                    meta["final_score"] = final

                doc = Document(page_content=text, metadata=meta)
                candidates.append((final, doc))

        # 全局排序 + 截取
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in candidates[:self.k]]

class MultiCollectionQdrantRetrieverMemory(BaseMemory):
    retriever: Any = Field()  # ✅ Pydantic 字段声明
    memory_key: str = "history"

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        query = inputs.get("input", "")
        docs = self.retriever.get_relevant_documents(query)
        combined = "\n\n".join(f"【{doc.metadata.get('title', '无标题')}】\n{doc.page_content}" for doc in docs)
        return {self.memory_key: combined}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        pass  # 禁止写入

    def clear(self) -> None:
        pass

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]


# ==== 初始化 ====
qdrant_path = "./Qdrant_Data/all-mpnet-base-v2_cosine_v2.0"
# qdrant_path = "./Qdrant_Data/bge-base-en-v1.5_cosine_v1.0_AES_7816_CPA"
qdrant_client = QdrantClient(path=qdrant_path)

# ==== 使用 HF 的 embedding 模型 维度768,这里模型要跟数据库匹配====
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

collections = qdrant_client.get_collections()
all_collections = [c.name for c in collections.collections]

# ★ 根据路径推断距离模式
distance_mode = infer_distance_mode_from_path(qdrant_path, default="cosine")
# print("Detected distance_mode =", distance_mode)

# 若是欧氏距离，阈值的语义是“最大距离”，不确定就先别设阈值
score_threshold = None



multi_retriever = MultiCollectionQdrantRetrieverLTM(
    qdrant_client=qdrant_client,
    collection_names=all_collections,
    embeddings=embeddings,
    top_k=1,
    score_threshold=score_threshold,         # euclid时语义为“最大距离”，不确定可 None
    per_collection_fetch=None,   # 默认 max(k,10)
    long_term_collection="long_term_memory",
    collection_boost=0.70,
    title_sim_weight=0.80,
    combine_mode="mul",          # "mul"更稳，"add"更激进
    distance_mode=distance_mode,      # 全库统一：cosine | dot | euclid
)

# Memory 包装保持不变
long_term_memory = MultiCollectionQdrantRetrieverMemory(retriever=multi_retriever)


# llm_model = "deepseek-ai/DeepSeek-V3.1-Terminus"
# llm_model = "zai-org/GLM-4.5"
# llm_model = "moonshotai/Kimi-K2-Instruct"

llm_model = "Pro/deepseek-ai/DeepSeek-V3.2"



llm = ChatOpenAI(
    model=llm_model,
    temperature=1,
    max_tokens=16383,
    # thinking_budget=32767,
    # messages = [
    #     {"role": "system", "content": "你是一个数据分析助手，专门用来处理和分析CSV数据文件。你会根据用户提供的指示进行数据分析、计算统计量和绘制图表。"}
    # ],
    openai_api_base="https://api.siliconflow.cn/v1",
    openai_api_key=""
)

# prompt = ChatPromptTemplate.from_template('''
# You are an expert assistant specialized in side-channel analysis.
#
# Here is relevant knowledge from your long-term memory (retrieved by semantic search based on current question):
# --------------------
# {history}
# --------------------
#
# Always try to incorporate relevant knowledge from memory into your reasoning and tool use. If the memory contains useful technical detail, use it to help guide tool selection or parameter choices.
#
# You can use the following tools:
#
# {tools}
#
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input:  the input to the action.
# - If you use the Jupyter API tool, your Action Input must be a Python code snippet, starting with ```python and ending with ```.
# - If you use the TimingAnalyzer tool, your Action Input must provide a single-line JSON input like {{\"npy_file\": \"path/to/file.npy\", \"num_traces\": 1000, \"num_points\": 1711}}.
# - If you use the TVLAAnalyzer tool, your Action Input must provide a single-line JSON input like {{\"npy_file\": \"path/to/file.npy\", \"num_traces\": 1000, \"num_points\": 1711}}.
# - If you use the RAG_retrieve_and_summarize tool, your Action Input must provide a single-line JSON input like {{\"keyword\": \"功耗攻击\"}}.
# Observation: the result of the action
# ... (this Thought/ Action/ Action Input/ Observation  can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
#
# Begin!
#
# Question: {input}
# {agent_scratchpad}
# ''')

# 不带采波工具
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template('''
# You are an expert assistant specialized in side-channel analysis.
#
# Here is relevant knowledge from your long-term memory (retrieved by semantic search based on current question):
# --------------------
# {history}
# --------------------
#
# Always try to incorporate relevant knowledge from memory into your reasoning and tool use. If the memory contains useful technical detail, use it to help guide tool selection or parameter choices.
#
# You can use the following tools:
#
# {tools}
#
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input:  the input to the action.
# - If you use the Jupyter API tool, your Action Input must be a Python code snippet, starting with ```python and ending with ```.
# - If you use the TimingAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_traces": 1000, "num_points": 1711}}.
# - If you use the TVLAAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_traces": 1000, "num_points": 1711}}.
# - If you use the RAG_retrieve_and_summarize tool, your Action Input must provide a single-line JSON input like {{"keyword": "功耗攻击"}}.
# Observation: the result of the action
# ... (this Thought/ Action/ Action Input/ Observation  can repeat N times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question
#
# Begin!
#     '''),
#     HumanMessagePromptTemplate.from_template("Question: {input}\n{agent_scratchpad}")
# ])



# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template('''
# You are an expert assistant specialized in side-channel analysis.
#
# Here is relevant knowledge from your long-term memory (retrieved by semantic search based on current question):
# --------------------
# {history}
# --------------------
#
# Always try to incorporate relevant knowledge from memory into your reasoning and tool use. If the memory contains useful technical detail, use it to help guide tool selection or parameter choices.
#
# You can use the following tools:
#
# {tools}
#
# Use the following format:
#
# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input:  the input to the action.
# - If you use the Jupyter API tool, your Action Input must be a Python code snippet, starting with ```python and ending with ```. When you generate plots, you MUST use `plt.show()` to display them.
# - If you use the TimingAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_traces": 1000, "num_points": 1711, "sampling_rate": 5000000, "clock_freq": 4800}}.
# - If you use the TVLAAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_traces": 1000, "num_points": 1711}}. This tool generates Python code for TVLA analysis. In your next step, you MUST use the JupyterAPITool to execute the returned code.
# - If you use the TraceCapture tool, your Action Input must provide a single-line JSON input like "json": {{\"npy_file\": \"path/to/file.npy\", \"point_num\": 240000, \"trace_num\": 1}}.
# - If you use the CPA tool, your Action Input must provide a single-line JSON input like "{{"plaintext_npy": "./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_plain.npy", "wave_npy": "./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave.npy", "candidates": 4, "leakage_model": "HW", "leakage_position": "S-box_out", "plot_path": "./cpa_trace_data/AES_128_7816/cpa_corr.png"}}.
# - If you use the Lowpass_Filter tool, your Action Input must provide a single-line JSON input like
# {{"wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave.npy","weight":10,"out_wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave_lowpass.npy","plot_path":"./cpa_trace_data/AES_128_7816/lowpass_compare.png","trace_index":0}}
# - If you use the Anchor_Picker tool, your Action Input must provide a single-line JSON input like
# {{"wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave_lowpass.npy","weight":10,"min_distance":300,"edge_margin":300,"default_win_ratio":0.025,"max_offset_ratio":0.5,"threshold":90}}
# - If you use the Static_Aligner tool, your Action Input must provide a single-line JSON input like
# {{"wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave_lowpass.npy","ref_trace":0,"start":836,"length":18,"max_shift":300,"threshold":90,"out_aligned":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave_lowpass_aligned.npy","plot_path":"./cpa_trace_data/AES_128_7816/align_overlay.png","show_indices":[0,1,2]}}
# Observation: the result of the action
# ... (this Thought/ Action/ Action Input/ Observation  can repeat N times)
#
# ---
# **IMPORTANT RULE: As soon as you have enough information to definitively answer the user's original question, you MUST stop using tools. Your very last output must strictly follow the format below:**
# ---
#
# Thought: I now have the final answer and will provide it.
# Final Answer: [your final, complete, and well-formatted answer to the original question here]
#
# Begin!
# '''),
#     HumanMessagePromptTemplate.from_template("{input}\n{agent_scratchpad}")
# ])




# tools = [JupyterAPITool(), TATool(), TVLATool(), SPATool(), RAG_retrieve_and_summarize_Tool(retriever_dict=retriever_dict)]      # TraceCaptureTool(),
# tools = [JupyterAPITool(), TATool(), TVLATool(), SPATool(), TraceCaptureTool(), RAG_retrieve_and_summarize_Tool(retriever_dict=retriever_dict)]
# tools = [JupyterAPITool(), TATool(), TVLATool(), TraceCaptureTool(), CPATool(), LowpassTool(), StaticAlignTool()]

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template('''
You are an expert assistant specialized in side-channel analysis.

Here is relevant knowledge from your long-term memory (retrieved by semantic search based on current question):
--------------------
{history}
--------------------

Always try to incorporate relevant knowledge from memory into your reasoning and tool use. If the memory contains useful technical detail, use it to help guide tool selection or parameter choices.

You can use the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input:  the input to the action.
- If you use the Jupyter API tool, your Action Input must be a Python code snippet, starting with ```python and ending with ```. When you generate plots, you MUST use `plt.show()` to display them.
- If you use the Lowpass_Filter tool, your Action Input must provide a single-line JSON input like
{{"wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave.npy","weight":10,"trace_index":0}}
- If you use the Anchor_Picker tool, your Action Input must provide a single-line JSON input like
{{"wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave_lowpass.npy","weight":10,"min_distance":600,"edge_margin":400,"default_win_ratio":0.03,"max_offset_ratio":0.2,"threshold":85}}
- If you use the Static_Aligner tool, your Action Input must provide a single-line JSON input like
{{"wave_npy":"./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave_lowpass.npy","ref_trace":0,"start":836,"length":18,"max_shift":300,"threshold":90,"show_indices":[0,1,2]}}
- If you use the CPA tool, your Action Input must provide a single-line JSON input like "{{"plaintext_npy": "./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_plain.npy", "wave_npy": "./cpa_trace_data/AES_128_7816/AES-128-7816-100traces-7400points_wave.npy", "candidates": 4, "leakage_model": "HW", "leakage_position": "S-box_out"}}.

Observation: the result of the action
... (this Thought/ Action/ Action Input/ Observation  can repeat N times)

---
**IMPORTANT RULE: As soon as you have enough information to definitively answer the user's original question, you MUST stop using tools. Your very last output must strictly follow the format below:**
---

Thought: I now have the final answer and will provide it.
Final Answer: [your final, complete, and well-formatted answer to the original question here]

Begin!
'''),
    HumanMessagePromptTemplate.from_template("{input}\n{agent_scratchpad}")
])


tools = [JupyterAPITool(), LowpassTool(),AnchorPickTool(),  StaticAlignTool(), CPATool()]

# Create the ReAct agent
agent = create_react_agent(
    llm,
    tools = tools,
    prompt=prompt,
)

# Thought → (Action → Action Input → Observation)* → Thought → Final Answer

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory = long_term_memory,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code = True,
    return_intermediate_steps=True,
    max_iterations=40,          # 最多循环40步
    early_stopping_method="generate",
    # max_execution_time=60,     # 最多运行60秒
    # output_parser=ReActSingleInputOutputParser(),  # 支持 Final Answer 判断
)


def inspect_prompt(query: str, agent_executor, memory, prompt, tools):
    """
    用于查看某个 query 会生成什么 prompt（包括工具、历史记忆、scratchpad）。
    不执行智能体动作，只返回 prompt 字符串。
    """

    try:
        # # 🧪 DEBUG 检查实际从 memory 检索到的内容（是否匹配成功）
        # docs = memory.retriever.get_relevant_documents(query)
        # for i, doc in enumerate(docs, 1):
        #     print(f"{i}. metadata: {doc.metadata}")
        # print(f"\n🔎 实际从 memory 中检索到 {len(docs)} 条相关记忆：\n")
        # for i, doc in enumerate(docs, 1):
        #     title = doc.metadata.get("title", "无")
        #     preview = doc.page_content[:150].replace("\n", " ")
        #     print(f"{i}. 【标题】{title}\n   内容预览：{preview}...\n")

        # 1️⃣ 获取从记忆中检索出来的历史（history）
        memory_vars = memory.load_memory_variables({"input": query})

        # 2️⃣ 构造工具和工具名
        tool_list = tools
        tool_names = ", ".join([tool.name for tool in tool_list])

        # 3️⃣ 构造 prompt 所需的输入字典
        prompt_inputs = {
            "input": query,
            "tools": tool_list,
            "tool_names": tool_names,
            "agent_scratchpad": "",  # 不执行，默认空
            **memory_vars
        }

        # 4️⃣ 格式化 prompt 为最终字符串
        formatted_prompt = prompt.format_prompt(**prompt_inputs).to_string()

        print("\n🧠 ===== Prompt Preview Start =====\n")
        print(formatted_prompt)
        print("\n🧠 ===== Prompt Preview End =====\n")

        return formatted_prompt

    except Exception as e:
        print(f"❌ 生成 prompt 失败: {e}")
        return None


# ==============================================================================
# 5. 全新的、简洁的 Markdown 日志函数
# ==============================================================================


# 假设 JupyterAPITool 类和 start_kernel_if_needed 函数已在代码的其他部分定义
# from your_module import JupyterAPITool, start_kernel_if_needed


# def run_agent_and_log_md(agent_executor, query, llm_model_name, prompt_text, log_path="agent_log.md"):
#     """
#     执行 Agent，并将所有信息记录到结构清晰、美观的 Markdown 文件中。
#     新版：优化了日志结构和代码块的显示。
#     """
#     kernel_proc = None
#     try:
#         # --- 动态准备与配置 (这部分不变) ---
#         log_base_name = log_path.rsplit('.', 1)[0]
#         image_dir_path = f"{log_base_name}_images"
#         for tool in agent_executor.tools:
#             if isinstance(tool, JupyterAPITool):
#                 tool.image_save_path = image_dir_path
#                 print(f"🔧 已动态配置 JupyterAPITool，图片将保存至: {image_dir_path}")
#                 break
#
#         kernel_proc = start_kernel_if_needed()
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         start_time = time.time()
#
#         # --- 文件头和摘要 (这部分不变) ---
#         header_content = [f"# Agent Execution Log", f"**Query:** `{query}`", f"**LLM Model:** `{llm_model_name}`",
#                           f"**Start Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`"]
#         body_content = []
#         if prompt_text:
#             body_content.extend(["\n---\n## Full Prompt to LLM\n", "<details><summary>Click to expand</summary>\n",
#                                  f"```\n{prompt_text}\n```", "\n</details>\n"])
#
#         # --- 执行核心任务 (这部分不变) ---
#         result = agent_executor.invoke({"input": query})
#
#         end_time = time.time()
#         duration = end_time - start_time
#         summary_content = [f"- **End Time:** `{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}`",
#                            f"- **Total Duration:** `{duration:.2f} seconds`"]
#
#         # ==================== 全新的、美化的日志记录逻辑 ====================
#         if result:
#             steps = result.get("intermediate_steps", [])
#             if steps:
#                 body_content.append("\n---\n## Reasoning Steps\n")
#                 for i, step in enumerate(steps, 1):
#                     action, observation = step
#
#                     # 使用 Markdown 引用块来清晰地组织每一步
#                     body_content.append(f"### Step {i}\n")
#                     body_content.append("> **Thought:**\n")
#                     body_content.append(f"> {action.log.strip()}\n")
#                     body_content.append("---")  # 分割线
#                     body_content.append(f"> **Action:** `{action.tool}`\n")
#
#                     # 智能判断 Action Input 的格式
#                     tool_input = action.tool_input
#                     if isinstance(tool_input, str) and tool_input.strip().startswith("```python"):
#                         # 如果是 Jupyter 代码，直接使用，不再嵌套
#                         body_content.append(f"> **Action Input:**\n{tool_input}\n")
#                     else:
#                         # 对于其他工具（如TVLA），使用 json 代码块
#                         body_content.append(f"> **Action Input:**\n> ```json\n> {tool_input}\n> ```\n")
#
#                     body_content.append("---")  # 分割线
#                     body_content.append(f"> **Observation:**\n> ```text\n{observation.strip()}\n> ```\n")
#
#             body_content.append("---\n## Final Answer\n")
#             body_content.append(result.get("output", "No final answer provided."))
#         # ====================================================================
#
#         final_content = header_content + ["\n## Execution Summary"] + summary_content + body_content
#
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(final_content))
#
#         print(f"✅ Agent 执行完成，Markdown 日志已保存到 {log_path}")
#         return result
#
#     except Exception as e:
#         # ... (错误处理不变) ...
#         print(f"❌ Agent 执行期间发生严重错误：{e}")
#         with open(log_path, "a", encoding="utf-8") as f:
#             f.write(f"\n\n---\n\n## AGENT EXECUTION FAILED\n\n```\n{e}\n```")
#         return None
#
#     finally:
#         # ... (资源清理不变) ...
#         if kernel_proc:
#             print("🧹 正在自动关闭后台内核服务...")
#             kernel_proc.terminate()
#             kernel_proc.wait()
#             print("✅ 内核服务已释放。")
#




# (确保在你的文件顶部导入 textwrap, 这是一个Python标准库)


# def run_agent_and_log_md(agent_executor, query, llm_model_name, prompt_text, log_path="agent_log.md"):
#     """
#     执行 Agent，并以最纯粹的“流水账”模式记录所有中间过程和最终结果。
#     这个版本不添加任何额外的格式、标题或清洗。
#     """
#     kernel_proc = None
#     try:
#         # --- 动态准备与配置 (这部分不变) ---
#         log_base_name = log_path.rsplit('.', 1)[0]
#         image_dir_path = f"{log_base_name}_images"
#         for tool in agent_executor.tools:
#             if isinstance(tool, JupyterAPITool):
#                 tool.image_save_path = image_dir_path
#                 print(f"🔧 已动态配置 JupyterAPITool，图片将保存至: {image_dir_path}")
#                 break
#
#         kernel_proc = start_kernel_if_needed()
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         start_time = time.time()
#
#         # --- 文件头和摘要 (这部分不变) ---
#         header_content = [f"# Agent Execution Log", f"**Query:** `{query}`", f"**LLM Model:** `{llm_model_name}`",
#                           f"**Start Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`"]
#         body_content = []
#         if prompt_text:
#             body_content.extend(["\n---\n## Full Prompt to LLM\n", "<details><summary>Click to expand</summary>\n",
#                                  f"```\n{prompt_text}\n```", "\n</details>\n"])
#
#         # --- 执行核心任务 (这部分不变) ---
#         result = agent_executor.invoke({"input": query})
#
#         end_time = time.time()
#         duration = end_time - start_time
#         summary_content = [f"- **End Time:** `{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}`",
#                            f"- **Total Duration:** `{duration:.2f} seconds`"]
#
#         # ==================== 全新的、纯粹的“流水账”日志记录逻辑 ====================
#         if result:
#             steps = result.get("intermediate_steps", [])
#             if steps:
#                 body_content.append("\n---\n## Execution Log\n")
#
#                 # 创建一个包含所有步骤原始文本的列表
#                 raw_log_parts = []
#                 for action, observation in steps:
#                     # 直接追加原始的、未经清洗的 action.log
#                     raw_log_parts.append(action.log.strip())
#
#                     # 对 observation 进行唯一的图片换行处理
#                     observation_text = str(observation).strip()
#                     observation_text = observation_text.replace(
#                         "![Generated Image]",
#                         "\n\n![Generated Image]"
#                     )
#                     raw_log_parts.append(observation_text)
#
#                 # 用双换行符将所有部分连接成一个连续的日志流
#                 body_content.append("\n\n".join(raw_log_parts))
#
#             # 记录最终答案
#             body_content.append("\n\n---\n## Final Answer\n")
#             body_content.append(result.get("output", "No final answer provided."))
#         # ====================================================================
#
#         final_content = header_content + ["\n## Execution Summary"] + summary_content + body_content
#
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(final_content))
#
#         print(f"✅ Agent 执行完成，Markdown 日志已保存到 {log_path}")
#         return result
#
#     except Exception as e:
#         # ... (错误处理不变) ...
#         print(f"❌ Agent 执行期间发生严重错误：{e}")
#         with open(log_path, "a", encoding="utf-8") as f:
#             f.write(f"\n\n---\n\n## AGENT EXECUTION FAILED\n\n```\n{e}\n```")
#         return None
#
#     finally:
#         # ... (资源清理不变) ...
#         if kernel_proc:
#             print("🧹 正在自动关闭后台内核服务...")
#             kernel_proc.terminate()
#             kernel_proc.wait()
#             print("✅ 内核服务已释放。")


# ==============================================================================
# 6. 主程序入口
# ==============================================================================

# def run_agent_and_log_md(agent_executor, query, llm_model_name, prompt_text, log_path="agent_log.md"):
#     """
#     执行 Agent，并以“绝对原始”的模式记录日志。
#     这个版本不解包 intermediate_steps，直接打印每一步的原始元组结构。
#     注意：此模式下图片链接不会被渲染。
#     """
#     kernel_proc = None
#     try:
#         # --- 动态准备与配置 (这部分不变) ---
#         log_base_name = log_path.rsplit('.', 1)[0]
#         image_dir_path = f"{log_base_name}_images"
#         for tool in agent_executor.tools:
#             if isinstance(tool, JupyterAPITool):
#                 tool.image_save_path = image_dir_path
#                 print(f"🔧 已动态配置 JupyterAPITool，图片将保存至: {image_dir_path}")
#                 break
#
#         kernel_proc = start_kernel_if_needed()
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         start_time = time.time()
#
#         # --- 文件头和摘要 (这部分不变) ---
#         header_content = [f"# Agent Execution Log", f"**Query:** `{query}`", f"**LLM Model:** `{llm_model_name}`",
#                           f"**Start Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`"]
#         body_content = []
#         if prompt_text:
#             body_content.extend(["\n---\n## Full Prompt to LLM\n", "<details><summary>Click to expand</summary>\n",
#                                  f"```\n{prompt_text}\n```", "\n</details>\n"])
#
#         # --- 执行核心任务 (这部分不变) ---
#         result = agent_executor.invoke({"input": query})
#
#         end_time = time.time()
#         duration = end_time - start_time
#         summary_content = [f"- **End Time:** `{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}`",
#                            f"- **Total Duration:** `{duration:.2f} seconds`"]
#
#         # ==================== 方案 A：绝对原始的日志记录逻辑 ====================
#         if result:
#             steps = result.get("intermediate_steps", [])
#             if steps:
#                 body_content.append("\n---\n## Execution Log (Raw Tuples)\n")
#
#                 # 循环时不解包，直接获取整个 step 元组
#                 for i, step in enumerate(steps, 1):
#                     # 将整个元组 (AgentAction, observation) 转换为它的字符串表示
#                     raw_step_string = str(step)
#
#                     # 为了可读性，我们仍然为每一步添加标题
#                     body_content.append(f"### Step {i}\n")
#                     # 将原始的元组字符串放入代码块，以保留所有格式
#                     body_content.append(f"```text\n{raw_step_string}\n```\n")
#
#             # 记录最终答案
#             body_content.append("\n\n---\n## Final Answer\n")
#             body_content.append(result.get("output", "No final answer provided."))
#         # ====================================================================
#
#         final_content = header_content + ["\n## Execution Summary"] + summary_content + body_content
#
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(final_content))
#
#         print(f"✅ Agent 执行完成，Markdown 日志已保存到 {log_path}")
#         return result
#
#     except Exception as e:
#         # ... (错误处理不变) ...
#         print(f"❌ Agent 执行期间发生严重错误：{e}")
#         with open(log_path, "a", encoding="utf-8") as f:
#             f.write(f"\n\n---\n\n## AGENT EXECUTION FAILED\n\n```\n{e}\n```")
#         return None
#
#     finally:
#         # ... (资源清理不变) ...
#         if kernel_proc:
#             print("🧹 正在自动关闭后台内核服务...")
#             kernel_proc.terminate()
#             kernel_proc.wait()
#             print("✅ 内核服务已释放。")


def stream_agent_and_log_md(agent_executor, query, llm_model_name, prompt_text, log_path="agent_log.md"):
    """
    使用 stream() 方法执行 Agent，并以最纯粹的模式记录每一个返回的原始 `chunk`。
    这个版本不做任何过滤和美化，只为让你观察最原始的数据结构。
    """
    kernel_proc = None
    try:
        # --- 动态准备与配置 (这部分不变) ---
        log_base_name = log_path.rsplit('.', 1)[0]
        image_dir_path = f"{log_base_name}_images"
        for tool in agent_executor.tools:
            if isinstance(tool, JupyterAPITool):
                tool.image_save_path = image_dir_path
                print(f"🔧 已动态配置 JupyterAPITool，图片将保存至: {image_dir_path}")
                break

        kernel_proc = start_kernel_if_needed()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        start_time = time.time()

        # --- 文件头 ---
        header_content = [f"# Agent Execution Log", f"**Query:** `{query}`", f"**LLM Model:** `{llm_model_name}`",
                          f"**Start Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`"]
        if prompt_text:
            header_content.extend(["\n---\n## Full Prompt to LLM\n", "<details><summary>Click to expand</summary>\n",
                                   f"```\n{prompt_text}\n```", "\n</details>\n"])

        # ==================== 纯粹的 Chunk 记录循环 ====================
        final_answer = ""
        # 使用 'a' (追加) 模式打开文件，因为我们会分部分写入
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_content))
            f.write("\n\n---\n## Raw Agent Stream Chunks\n")

            # 使用 enumerate 来为每个 chunk 编号，方便观察
            for i, chunk in enumerate(agent_executor.stream({"input": query})):

                # 将整个 chunk 字典转换为字符串表示形式
                # str() 是一个安全的方式，可以表示包含复杂对象的字典
                chunk_representation = str(chunk)

                # 写入 chunk 编号和其原始内容
                f.write(f"\n### --- Chunk {i + 1} ---\n")
                f.write(f"```text\n{chunk_representation}\n```\n")
                f.flush()  # 实时刷新到文件

                # 我们仍然需要从流中捕获最终答案，以便在日志末尾总结
                if "output" in chunk:
                    final_answer = chunk.get("output", "")
        # =============================================================

        # --- 循环结束后，记录最终答案和摘要 ---
        end_time = time.time()
        duration = end_time - start_time
        summary_content = [f"\n---\n## Execution Summary\n",
                           f"- **End Time:** `{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}`",
                           f"- **Total Duration:** `{duration:.2f} seconds`"]
        final_answer_content = [f"\n\n---\n## Final Answer\n", final_answer]

        # 使用 'a' (追加) 模式将最后的信息写入文件
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(summary_content))
            f.write("\n".join(final_answer_content))

        print(f"✅ Agent 执行完成，Markdown 日志已保存到 {log_path}")
        return {"output": final_answer}

    except Exception as e:
        # ... (错误处理不变) ...
        print(f"❌ Agent 执行期间发生严重错误：{e}")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n## AGENT EXECUTION FAILED\n\n```\n{e}\n```")
        return None

    finally:
        # ... (资源清理不变) ...
        if kernel_proc:
            print("🧹 正在自动关闭后台内核服务...")
            kernel_proc.terminate()
            kernel_proc.wait()
            print("✅ 内核服务已释放。")


# def run_agent_and_log_md(agent_executor, query, llm, prompt_text, log_path="agent_log.md"):
#     """
#     通过捕获 LangChain 的 verbose 标准输出，生成最理想、最完整的流水账日志。
#     这是包含完整错误处理和资源清理的最终版本。
#     """
#     kernel_proc = None
#     try:
#         # --- 1. 动态准备与配置 ---
#         log_base_name = log_path.rsplit('.', 1)[0]
#         image_dir_path = f"{log_base_name}_images"
#         for tool in agent_executor.tools:
#             if isinstance(tool, JupyterAPITool):
#                 tool.image_save_path = image_dir_path
#                 print(f"🔧 已动态配置 JupyterAPITool，图片将保存至: {image_dir_path}")
#                 break
#
#         kernel_proc = start_kernel_if_needed()
#         os.makedirs(os.path.dirname(log_path), exist_ok=True)
#         start_time = time.time()
#
#         # --- 2. 捕获 Agent 的完整执行日志 ---
#         log_capture_buffer = io.StringIO()
#         with redirect_stdout(log_capture_buffer):
#             result = agent_executor.invoke({"input": query})
#         verbose_log_content = log_capture_buffer.getvalue()
#
#         end_time = time.time()
#         duration = end_time - start_time
#
#         # --- 3. 清洗和准备日志内容 ---
#         ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
#         clean_log_content = ansi_escape_pattern.sub('', verbose_log_content)
#
#         # --- 4. 准备文件所有部分的内容 ---
#         # header_content = [f"# Agent Execution Log", f"**Query:** `{query}`"]
#         header_content = [
#             f"# Agent Execution Log",
#             f"**Query:**",
#             f"```text\n{query}\n```"
#         ]
#
#         summary_content = [
#             "\n---\n## Execution Summary\n",
#             f"- **LLM Model:** `{getattr(llm, 'model_name', 'N/A')}`",
#             f"- **Start Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`",
#             f"- **End Time:** `{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}`",
#             f"- **Total Duration:** `{duration:.2f} seconds`"
#         ]
#
#         llm_settings_content = [
#             "\n### LLM Settings",
#             f"- **Temperature:** `{getattr(llm, 'temperature', 'N/A')}`",
#             f"- **Max Tokens:** `{getattr(llm, 'max_tokens', 'N/A')}`"
#         ]
#
#         tool_list_str = "\n".join([f"- {tool.name}" for tool in agent_executor.tools])
#         tools_content = ["\n### Tools Used", tool_list_str]
#
#         prompt_content = []
#         if prompt_text:
#             prompt_content.extend(["\n---\n## Full Prompt to LLM\n", "<details><summary>Click to expand</summary>\n",
#                                    f"```\n{prompt_text}\n```", "\n</details>\n"])
#
#         # ==================== 最终的“整体替换”日志记录逻辑 ====================
#         # 1. 首先，将清洗后的所有日志内容，统一包裹在一个大的 ```text 代码块中
#         execution_log_body = f"```text\n{clean_log_content.strip()}\n```"
#
#         # 2. 定义图片链接的正则表达式
#         image_pattern = re.compile(r"(!\[Generated Image\]\(.*?\))")
#
#         # 3. 使用 re.sub 进行查找和替换，将图片链接从代码块中“拯救”出来
#         # 替换的逻辑是：在图片链接前加上“结束代码块”和换行，在图片链接后加上“开始代码块”
#         # 这样就在图片出现的位置，精准地打断并重开了代码块
#         execution_log_body = image_pattern.sub(r"```\n\n\1\n\n```text", execution_log_body)
#
#         execution_log_content = ["\n---\n## Execution Log\n", execution_log_body]
#         # ====================================================================
#
#         # ouput = result.get("output", "").strip()
#         # final_answer_content = [f"**Output:** `{ouput}`"]
#
#         output_text = result.get("output", "").strip()
#         final_answer_content = [
#             f"\n\n---\n**Output:**\n",
#             f"```text\n{output_text}\n```"
#         ]
#
#         # --- 5. 拼接所有内容并写入文件 ---
#         final_content = (
#                 header_content + summary_content + llm_settings_content +
#                 tools_content + prompt_content + execution_log_content + final_answer_content
#         )
#
#         with open(log_path, "w", encoding="utf-8") as f:
#             f.write("\n".join(final_content))
#
#         print(f"✅ Agent 执行完成，Markdown 日志已保存到 {log_path}")
#         return result
#
#     except Exception as e:
#         # 完整的错误处理逻辑
#         print(f"❌ Agent 执行期间发生严重错误：{e}")
#         # 确保日志目录存在，以防在早期阶段出错
#         if log_path and os.path.dirname(log_path):
#             os.makedirs(os.path.dirname(log_path), exist_ok=True)
#             # 以追加模式打开日志文件，记录下错误信息
#             with open(log_path, "a", encoding="utf-8") as f:
#                 f.write(f"\n\n---\n\n## AGENT EXECUTION FAILED\n\n")
#                 f.write(f"```\n{e}\n```")
#         return None
#
#     finally:
#         # 完整的资源清理逻辑
#         if kernel_proc:
#             print("🧹 正在自动关闭后台内核服务...")
#             kernel_proc.terminate()  # 终止子进程
#             kernel_proc.wait()  # 等待进程完全关闭，防止僵尸进程
#             print("✅ 内核服务已释放。")

def run_agent_and_log_md(agent_executor, query, llm, prompt_text, log_path="log/agent_log_PLACEHOLDER/agent_log.md"):
    """
    通过捕获 LangChain 的 verbose 标准输出，生成最理想、最完整的流水账日志。
    统一“本次运行”的 run 目录，使本次运行期间生成的所有文件（图片、npy、显式保存的图等）
    都直接保存在该 run 目录下（无子目录）。
    """
    kernel_proc = None
    try:
        if not log_path:
            raise ValueError("log_path 不能为空")
        if log_path.lower().endswith(".md"):
            run_dir = os.path.dirname(log_path) or "."
        else:
            run_dir = log_path  # 如果传的是目录，则直接用它
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "agent_log.md")  # 固定日志名

        # ================================
        # [MOD] 注入 run_dir 到各个工具里
        # ================================
        os.environ["AGENT_RUN_DIR"] = run_dir

        for tool in agent_executor.tools:
            # JupyterAPITool 的图片捕获目录 = run_dir
            if getattr(tool, "name", "") == "JupyterAPITool":
                # 工具类上通常定义了 image_save_path 属性
                setattr(tool, "image_save_path", run_dir)  # [MOD]
                print(f"🔧 Jupyter 捕获图目录: {run_dir}")
            # 会写文件的工具：统一注入 run_dir
            if hasattr(tool, "set_run_dir"):
                tool.set_run_dir(run_dir)  # [MOD]

        # 启动(或复用)本地内核服务
        kernel_proc = start_kernel_if_needed()  # 你现有的函数
        start_time = time.time()

        # ========== 执行，捕获LangChain verbose 输出 ==========
        log_capture_buffer = io.StringIO()
        with redirect_stdout(log_capture_buffer):
            result = agent_executor.invoke({"input": query})
        verbose_log_content = log_capture_buffer.getvalue()
        end_time = time.time()
        duration = end_time - start_time

        # ========== 清洗 ANSI ==========
        ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_log_content = ansi_escape_pattern.sub('', verbose_log_content)

        # ========== 组装 Markdown ==========
        header_content = [
            f"# Agent Execution Log",
            f"**Query:**",
            f"```text\n{query}\n```"
        ]

        summary_content = [
            "\n---\n## Execution Summary\n",
            f"- **LLM Model:** `{getattr(llm, 'model_name', 'N/A')}`",
            f"- **Start Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- **End Time:** `{datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- **Total Duration:** `{duration:.2f} seconds`",
            f"- **Run Directory:** `{run_dir}`"  # [MOD] 记录本次run目录
        ]

        llm_settings_content = [
            "\n### LLM Settings",
            f"- **Temperature:** `{getattr(llm, 'temperature', 'N/A')}`",
            f"- **Max Tokens:** `{getattr(llm, 'max_tokens', 'N/A')}`"
        ]

        tool_list_str = "\n".join([f"- {tool.name}" for tool in agent_executor.tools])
        tools_content = ["\n### Tools Used", tool_list_str]

        prompt_content = []
        if prompt_text:
            prompt_content.extend([
                "\n---\n## Full Prompt to LLM\n",
                "<details><summary>Click to expand</summary>\n",
                f"```\n{prompt_text}\n```",
                "\n</details>\n"
            ])

        # --- 执行日志主体，保留图片中断逻辑 ---
        execution_log_body = f"```text\n{clean_log_content.strip()}\n```"
        image_pattern = re.compile(r"(!\[Generated Image\]\(.*?\))")
        execution_log_body = image_pattern.sub(r"```\n\n\1\n\n```text", execution_log_body)
        execution_log_content = ["\n---\n## Execution Log\n", execution_log_body]

        output_text = result.get("output", "").strip()
        final_answer_content = [
            f"\n\n---\n**Output:**\n",
            f"```text\n{output_text}\n```"
        ]

        final_content = (
            header_content + summary_content + llm_settings_content +
            tools_content + prompt_content + execution_log_content + final_answer_content
        )

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(final_content))

        print(f"✅ Agent 执行完成，Markdown 日志已保存到 {log_path}")
        return result

    except Exception as e:
        print(f"❌ Agent 执行期间发生错误：{e}")
        if log_path and os.path.dirname(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n---\n\n## AGENT EXECUTION FAILED\n\n")
                f.write(f"```\n{e}\n```")
        return None

    finally:
        if kernel_proc:
            print("🧹 正在自动关闭后台内核服务...")
            kernel_proc.terminate()
            kernel_proc.wait()
            print("✅ 内核服务已释放。")

def build_agent() -> Tuple[AgentExecutor, object]:
    """
    返回:
      - agent_executor: 你的ReAct Agent执行器（带 tools/memory/prompt）
      - llm: 供记录日志展示模型名/温度等信息的llm对象
    """

    def start_kernel_if_needed(script_path="python_jupyter_kernel_tool.py", port=5000):
        """
        检查端口，如果内核服务未运行，则在后台启动它。
        返回子进程对象，以便后续可以关闭它。
        """
        # 检查端口是否被占用
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) == 0:
                print(f"⚠️ 端口 {port} 已被占用，假设内核服务已由外部启动，跳过。")
                return None  # 返回 None 表示我们没有启动它

        print(f"🚀 正在后台启动内核服务 ({script_path})...")
        # 使用 Popen 在后台启动服务，并将输出重定向，避免干扰主程序
        kernel_proc = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 等待几秒钟，确保 Flask 服务有足够的时间完成初始化
        print("⏳ 等待内核服务初始化...")
        time.sleep(4)  # 这个等待时间很重要
        print("✅ 内核服务已启动。")

        return kernel_proc


    # 如需结构化入参，可启用此模型并在类中设置 args_schema = JupyterNotebookInput
    # class JupyterNotebookInput(BaseModel):
    #     code: str = Field(..., description="Python code to execute in Jupyter Notebook.")

    class JupyterAPITool(BaseTool):
        """
        终极稳健版 Jupyter 工具：
        1) 预处理：剥离```python外壳、统一换行、去整体缩进（dedent），根治“unexpected indent”；
        2) 注入：按 plt.show() 所在行的缩进注入图像捕获块；如用户无 show()，末尾顶格补一次；
        3) 安全：注释掉用户自写的 plt.savefig(...)，避免重复保存/冲突；
        4) 输出：从执行返回中抓取 Base64 图片，保存到 image_save_path，并替换为 Markdown 链接；
        5) 兼容：保存路径统一转为正斜杠，兼容前端渲染；其余文本/错误原样拼接返回。
        """

        name: str = "JupyterAPITool"
        description: str = (
            "Executes Python code in a Jupyter notebook environment. "
            "It returns text output and automatically saves any generated plots, embedding them as Markdown image links."
        )

        # 若需结构化入参，请取消下一行注释
        # args_schema: Type[BaseModel] = JupyterNotebookInput

        # 输出图片保存目录（相对或绝对）。建议由上层在运行前动态覆盖为 runs/<run_id>/images
        image_save_path: str = "log/default_images"

        # ===================== 内部辅助：代码预处理/注入 =====================

        @staticmethod
        def _preprocess_code(raw: str) -> str:
            """
            去掉三引号外壳、统一换行、去 BOM/不可见空格、整体去缩进。
            解决 LLM/Markdown 带来的隐形缩进与包裹问题。
            """
            s = raw.strip()
            # 去掉 ```python / ```py / ``` 头尾
            s = re.sub(r"^\s*```(?:python|py)?\s*\n?", "", s, flags=re.IGNORECASE)
            s = re.sub(r"\n?\s*```\s*$", "", s, flags=re.IGNORECASE)
            # 统一换行 & 清除 BOM/不可见空格
            s = s.replace("\r\n", "\n").replace("\r", "\n")
            s = s.lstrip("\ufeff").replace("\u00A0", " ")
            # 整体去缩进
            s = textwrap.dedent(s)
            # 去除首尾空行
            lines = s.split("\n")
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            return "\n".join(lines)

        @staticmethod
        def _indent_block(block: str, indent: str) -> str:
            """将多行 block 逐行加上指定缩进。空行保持空。"""
            return "\n".join((indent + ln if ln else ln) for ln in block.splitlines())

        @staticmethod
        def _capture_block() -> str:
            """返回用于替换 plt.show() 的无缩进图像捕获块（保持无前导空格，后续按需缩进）。"""
            return textwrap.dedent(
                """\
                import io, base64
                try:
                    import matplotlib.pyplot as plt
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    print(f"---IMAGE_BASE64_START---{img_b64}---IMAGE_BASE64_END---")
                except Exception as e:
                    print(f"Image capture failed: {e}")
                """
            ).rstrip("\n")

        @classmethod
        def _inject_capture_block(cls, code: str) -> str:
            """
            1) 注释掉显式 plt.savefig(...)；
            2) 将第一处 plt.show() 按其行缩进替换为图像捕获块；
            3) 若不存在 plt.show()，在末尾顶格补一个捕获块。
            """
            # 注释掉显式 savefig，避免重复保存
            code = re.sub(
                r"(?m)^\s*plt\.savefig\([^)]*\)\s*$",
                "# plt.savefig() was commented out by the tool.",
                code,
            )

            pat = re.compile(r"(?m)^(?P<indent>\s*)plt\.show\(\)\s*$")
            block = cls._capture_block()

            def repl(m: re.Match) -> str:
                indent = m.group("indent") or ""
                return cls._indent_block(block, indent)

            new_code, n = pat.subn(repl, code, count=1)
            if n == 0:
                # 用户没写 show()，末尾顶格补一次
                new_code = code.rstrip() + "\n\n" + block + "\n"
            return new_code

        # =========================== 主执行入口 ============================

        def _run(self, code: str) -> str:
            """
            传入一段 Python 代码（可带```python外壳），在本地 Jupyter 执行，抓取文本输出和图像并返回 Markdown。
            """
            save_dir = self.image_save_path
            os.makedirs(save_dir, exist_ok=True)

            # 1) 预处理：剥壳/换行/BOM/不可见空格/整体去缩进
            code = self._preprocess_code(code)

            # 2) 注入：按 plt.show() 行缩进替换图像捕获块；无 show() 则末尾补一个
            code = self._inject_capture_block(code)

            print("🔹 正在发送代码到 Jupyter API (主动捕获模式)...")

            try:
                # 3) 调用你的 Jupyter 执行服务
                resp = requests.post("http://localhost:5000/execute", json={"code": code})
                if resp.status_code != 200:
                    return f"❌ Jupyter API Error: {resp.status_code} - {resp.text}"

                # 4) 汇总文本输出 & 错误
                full_text_output = ""
                outputs = resp.json().get("outputs", [])
                for item in outputs:
                    if item.get("type") == "text":
                        full_text_output += item.get("content", "")
                    elif item.get("type") == "error":
                        full_text_output += f"\n---ERROR---\n{item.get('content', '')}"

                # 5) 处理 Base64 图片 → 文件 → Markdown
                start_marker = "---IMAGE_BASE64_START---"
                end_marker = "---IMAGE_BASE64_END---"
                pattern = re.compile(
                    f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}",
                    re.DOTALL,
                )

                def save_and_replace(match: re.Match) -> str:
                    img_b64 = match.group(1).strip()
                    try:
                        image_data = base64.b64decode(img_b64)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        image_name = f"img_{ts}.png"
                        image_full_path = os.path.join(save_dir, image_name)
                        with open(image_full_path, "wb") as f:
                            f.write(image_data)

                        # 统一正斜杠，前端 Markdown 更稳
                        rel = f"{os.path.basename(save_dir)}/{image_name}".replace("\\", "/")
                        # rel = image_name
                        print(f"🖼️ 图片已通过内存捕获并保存到 {image_full_path}")
                        return f"![Generated Image]({rel})"
                    except Exception as e:
                        print(f"❌ 图片解码或保存失败: {e}")
                        return f"[图片保存失败: {e}]"

                final_observation = pattern.sub(save_and_replace, full_text_output)

                print("✅ Jupyter 执行完成，已处理所有输出。")
                return final_observation.strip()

            except Exception as e:
                err = f"❌ Jupyter 工具发生未知错误: {e}"
                print(err)
                return err

    # 定义输入数据
    class TAInput(BaseModel):
        json: str = Field(...,
                          description="A JSON string containing npy_file, num_traces, num_points, sampling_rate and clock_freq.")

    # 定义 Timing Analysis 工具类
    class TATool(BaseTool):
        name: str = "TimingAnalyzer"
        description: str = (
            "执行 Timing Analysis（TA）测试，返回固定输入与随机输入平均穿越阈值时间之差 |T1 - T2|（单位：秒）。\n"
            "输入格式为 JSON 字符串，必须包含 npy_file, num_traces, num_points, sampling_rate，"
            "可选 voltage_threshold（未提供则自动计算）。"
        )
        args_schema: Type[BaseModel] = TAInput

        def _run(self, json_input: str) -> str:
            import json
            try:
                data = json.loads(json_input)
                npy_file = data["npy_file"]
                num_traces = data["num_traces"]
                num_points = data["num_points"]
                sampling_rate = data["sampling_rate"]
                clock_freq = data["clock_freq"]
                voltage_threshold = data.get("voltage_threshold", None)
            except Exception as e:
                return f"[TimingAnalyzer] Parameter resolution failed：{e}"

            code = f"""
    from timing_analysis_tool import compute_timing_difference

    delta = compute_timing_difference(
        npy_file="{npy_file}",
        num_traces={num_traces},
        num_points={num_points},
        sampling_rate={sampling_rate},
        clock_freq={clock_freq},
        {"voltage_threshold=" + str(voltage_threshold) + "," if voltage_threshold is not None else ""}
    )

    """

            return JupyterAPITool().run(code)

    class TVLAInput(BaseModel):
        # 使用清晰的名称 json_input，不再使用别名
        json_input: str = Field(..., description="一个包含 npy_file, num_traces, num_points 的 JSON 字符串")

    class TVLATool(BaseTool):
        name: str = "TVLAAnalyzer"
        description: str = "输入参数为：npy 文件、波形条数、点数。根据输入的参数，生成一段用于执行 TVLA 分析和绘图的 Python 代码。你应该在下一步中使用 JupyterAPITool 来执行这段返回的代码。"
        args_schema: Type[BaseModel] = TVLAInput

        # _run 方法的参数名与 Pydantic 模型中的字段名完全一致
        def _run(self, json_input: str) -> str:
            try:
                data = json.loads(json_input)
                # ... (后续代码不变)
                npy_file = data["npy_file"]
                num_traces = data["num_traces"]
                num_points = data["num_points"]
            except Exception as e:
                # 如果参数解析失败，返回一段注释代码，告知错误
                return f"# [TVLAAnalyzer] 参数解析失败：{e}"

            code = f"""
    from tvla_tool import compute_tvla_general
    import matplotlib.pyplot as plt
    t_values = compute_tvla_general("{npy_file}", {num_traces}, {num_points})
    # Add plotting
    plt.figure(figsize=(12, 6))
    plt.plot(t_values, 'b-', label='TVLA Result')
    plt.axhline(y=4.5, color='r', linestyle='--', label='Threshold (4.5)')
    plt.xlabel('Point Index')
    plt.ylabel('t-value')
    plt.title('TVLA Analysis Result')
    plt.legend()
    plt.show()

    # 打印结果
    print("t_values[:10]:", t_values[:10])
    print("Max:", max(t_values))
    print("Min:", min(t_values))
    """
            return code

    class TraceCaptureInput(BaseModel):
        json: str = Field(..., description="一个包含 npy_file, point_num, trace_num 的 JSON 字符串")

    # 定义TraceCapture工具，继承 BaseTool
    # 帮我采集1条智能卡上AES加密的波形，24万点。
    # - If you use the TraceCapture tool, your Action Input must provide a single-line JSON input like "json": {{\"npy_file\": \"path/to/file.npy\", \"point_num\": 240000, \"trace_num\": 1}}.
    class TraceCaptureTool(BaseTool):
        name: str = "TraceCapture"
        description: str = (
            "Capture traces and save to file. "
        )
        args_schema: Type[BaseModel] = TraceCaptureInput

        def _run(self, json_input: str) -> str:
            try:
                data = json.loads(json_input)
                npy_file = data["npy_file"]
                point_num = data["point_num"]
                trace_num = data["trace_num"]
            except Exception as e:
                return f"[TraceCapture] 参数解析失败：{e}"

            code = f"""
    from trace_capture_tool import trace_capture

    # 执行采集
    trace_capture("{npy_file}", point_num={point_num}, trace_num={trace_num})

    print("Trace captured and saved to:", "{npy_file}")
    """
            return JupyterAPITool().run(code)

    def _cosine(a, b) -> float:
        va = np.array(a);
        vb = np.array(b)
        na = np.linalg.norm(va);
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def _normalize_score(raw_score: float, distance_mode: str) -> float:
        """
        统一成“越大越好”的相似度：
          - cosine/dot: 直接返回 raw_score
          - euclid:     相似度 = 1/(1+d)
        """
        dm = (distance_mode or "cosine").lower()
        if dm in ("euclid", "euclidean", "l2"):
            d = max(float(raw_score), 0.0)
            return 1.0 / (1.0 + d)
        return float(raw_score)

    def infer_distance_mode_from_path(db_path: str, default: str = "cosine") -> str:
        """
        从数据库路径名里推断距离度量（大小写不敏感）。
        支持: "cosine" | "dot" | "euclid"/"euclidean"/"l2"
        """
        s = (db_path or "").lower()

        # 先匹配最明确的关键词
        if re.search(r"(?:^|[_\-./])cosine(?:$|[_\-./])", s):
            return "cosine"
        if re.search(r"(?:^|[_\-./])dot(?:$|[_\-./])", s):
            return "dot"
        if re.search(r"(?:^|[_\-./])(euclid|euclidean|l2)(?:$|[_\-./])", s):
            return "euclid"

        # 没有命中就回退默认
        return default

    class MultiCollectionQdrantRetrieverLTM:
        """
        仅对 long_term_collection 应用：标题相似度权重 + 集合权重。
        其他 collection 仅用文本的 base 相似度（Qdrant score 归一后）。
        """

        def __init__(
                self,
                qdrant_client: QdrantClient,
                collection_names: List[str],
                embeddings,
                *,
                top_k: int = 5,
                score_threshold: Optional[float] = None,  # euclid 下表示“最大距离”，不确定可 None
                per_collection_fetch: Optional[int] = None,  # 每库抓取多少候选再合并排序
                long_term_collection: str = "long_term_memory",
                collection_boost: float = 0.0,  # 仅对 long_term_collection 生效，如 0.35
                title_sim_weight: float = 0.30,  # 仅对 long_term_collection 生效
                combine_mode: str = "mul",  # "mul"（稳）或 "add"（激进）
                distance_mode: str = "cosine",  # 全局距离："cosine"|"dot"|"euclid"
        ):
            self.client = qdrant_client
            self.collections = collection_names if isinstance(collection_names, list) else [collection_names]
            self.embeddings = embeddings
            self.k = top_k
            self.score_threshold = score_threshold
            self.per_collection_fetch = per_collection_fetch
            self.long_term_collection = long_term_collection
            self.collection_boost = float(collection_boost)
            self.title_sim_weight = float(title_sim_weight)
            self.combine_mode = combine_mode
            self.distance_mode = distance_mode

            # 标题向量缓存，避免重复计算
            self._title_vec_cache: Dict[str, List[float]] = {}

        def _final_score(
                self,
                *,
                base_sim: float,
                is_ltm: bool,
                title_sim: Optional[float] = None
        ) -> float:
            """
            乘法融合（默认）：
                final = base_sim * (1 + collection_boost) * (1 + title_sim_weight * title_sim)
            加法融合：
                final = base_sim + collection_boost + title_sim_weight * title_sim
            仅 is_ltm=True 时才启用 collection_boost 与 title_sim。
            """
            if not is_ltm:
                return base_sim

            t = float(title_sim or 0.0)
            if self.combine_mode == "add":
                return base_sim + self.collection_boost + self.title_sim_weight * t
            else:  # "mul"
                return base_sim * (1.0 + self.collection_boost) * (1.0 + self.title_sim_weight * t)

        def get_relevant_documents(self, query: str) -> List[Document]:
            qvec = self.embeddings.embed_query(query)
            per_k = self.per_collection_fetch if self.per_collection_fetch else max(self.k, 10)

            candidates: List[Tuple[float, Document]] = []

            for col in self.collections:
                kwargs = dict(
                    collection_name=col,
                    query_vector=qvec,
                    limit=per_k,
                    with_payload=True,
                )
                if self.score_threshold is not None:
                    kwargs["score_threshold"] = self.score_threshold

                results = self.client.search(**kwargs)
                is_ltm = (col == self.long_term_collection)

                for pt in results:
                    payload: Dict[str, Any] = pt.payload or {}
                    title = payload.get("title", "无标题")
                    text = payload.get("text", "[无内容]")
                    pid = getattr(pt, "id", None)
                    raw_score = float(getattr(pt, "score", 0.0))

                    # 1) 归一基础分
                    base_sim = _normalize_score(raw_score, self.distance_mode)

                    # 2) 仅 LTM 计算标题相似度并融合
                    title_sim = None
                    if is_ltm:
                        if title in self._title_vec_cache:
                            tvec = self._title_vec_cache[title]
                        else:
                            tvec = self.embeddings.embed_query(title)
                            self._title_vec_cache[title] = tvec
                        title_sim = _cosine(qvec, tvec)

                    final = self._final_score(base_sim=base_sim, is_ltm=is_ltm, title_sim=title_sim)

                    # 3) 构造 Document（保留调参信息，便于 debug）
                    meta = dict(payload)
                    meta.setdefault("title", title)
                    meta["collection"] = col
                    meta["qdrant_id"] = pid
                    meta["raw_score"] = raw_score
                    meta["base_sim"] = base_sim
                    if is_ltm:
                        meta["title_sim"] = title_sim
                        meta["final_score"] = final

                    doc = Document(page_content=text, metadata=meta)
                    candidates.append((final, doc))

            # 全局排序 + 截取
            candidates.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in candidates[:self.k]]

    class MultiCollectionQdrantRetrieverMemory(BaseMemory):
        retriever: Any = Field()  # ✅ Pydantic 字段声明
        memory_key: str = "history"

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
            query = inputs.get("input", "")
            docs = self.retriever.get_relevant_documents(query)
            combined = "\n\n".join(f"【{doc.metadata.get('title', '无标题')}】\n{doc.page_content}" for doc in docs)
            return {self.memory_key: combined}

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
            pass  # 禁止写入

        def clear(self) -> None:
            pass

        @property
        def memory_variables(self) -> List[str]:
            return [self.memory_key]

    # ==== 初始化 ====
    # qdrant_path = "./Qdrant_Data/all-mpnet-base-v2_cosine_v2.0"
    qdrant_path = "./Qdrant_Data_api_copy/bge-base-en-v1.5_cosine_v1.0"
    qdrant_client = QdrantClient(path=qdrant_path)

    # ==== 使用 HF 的 embedding 模型 维度768,这里模型要跟数据库匹配====
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    collections = qdrant_client.get_collections()
    all_collections = [c.name for c in collections.collections]

    # ★ 根据路径推断距离模式
    distance_mode = infer_distance_mode_from_path(qdrant_path, default="cosine")
    # print("Detected distance_mode =", distance_mode)

    # 若是欧氏距离，阈值的语义是“最大距离”，不确定就先别设阈值
    score_threshold = None
    # 如果你很确定要加阈值，可按欧氏距离经验值设置，如：
    # if distance_mode == "euclid":
    #     score_threshold = 0.8  # 距离上限（示例值，按需调）

    multi_retriever = MultiCollectionQdrantRetrieverLTM(
        qdrant_client=qdrant_client,
        collection_names=all_collections,
        embeddings=embeddings,
        top_k=4,
        score_threshold=score_threshold,  # euclid时语义为“最大距离”，不确定可 None
        per_collection_fetch=None,  # 默认 max(k,10)
        long_term_collection="long_term_memory",
        collection_boost=0.70,  # 仅对 long_term_memory 生效
        title_sim_weight=0.80,  # 仅对 long_term_memory 生效
        combine_mode="mul",  # "mul"更稳，"add"更激进
        distance_mode=distance_mode,  # 全库统一：cosine | dot | euclid
    )

    # Memory 包装保持不变
    long_term_memory = MultiCollectionQdrantRetrieverMemory(retriever=multi_retriever)

    llm_model = "deepseek-ai/DeepSeek-V3.1-Terminus"
    # llm_model = "zai-org/GLM-4.6"
    # llm_model = "moonshotai/Kimi-K2-Instruct"

    # llm_model = "deepseek-ai/DeepSeek-V3"
    # llm_model = "deepseek-ai/DeepSeek-V2.5"
    # llm_model = "Qwen/Qwen3-8B"
    # llm_model = "Qwen/Qwen3-32B"
    # llm_model = "MiniMaxAI/MiniMax-M1-80k"
    # llm_model = "deepseek-ai/DeepSeek-R1"

    llm = ChatOpenAI(
        model=llm_model,
        temperature=1,
        max_tokens=16383,
        # thinking_budget=32767,
        # messages = [
        #     {"role": "system", "content": "你是一个数据分析助手，专门用来处理和分析CSV数据文件。你会根据用户提供的指示进行数据分析、计算统计量和绘制图表。"}
        # ],
        openai_api_base="https://api.siliconflow.cn/v1",
        openai_api_key=""
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template('''
    You are an expert assistant specialized in side-channel analysis.

    Here is relevant knowledge from your long-term memory (retrieved by semantic search based on current question):
    --------------------
    {history}
    --------------------

    Always try to incorporate relevant knowledge from memory into your reasoning and tool use. If the memory contains useful technical detail, use it to help guide tool selection or parameter choices.

    You can use the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input:  the input to the action.
    - If you use the Jupyter API tool, your Action Input must be a Python code snippet, starting with ```python and ending with ```. When you generate plots, you MUST use `plt.show()` to display them.
    - If you use the TimingAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_traces": 1000, "num_points": 1711, "sampling_rate": 5000000, "clock_freq": 4800}}.
    - If you use the TVLAAnalyzer tool, your Action Input must provide a single-line JSON input like {{"npy_file": "path/to/file.npy", "num_traces": 1000, "num_points": 1711}}. This tool generates Python code for TVLA analysis. In your next step, you MUST use the JupyterAPITool to execute the returned code.
    - If you use the TraceCapture tool, your Action Input must provide a single-line JSON input like "json": {{\"npy_file\": \"path/to/file.npy\", \"point_num\": 240000, \"trace_num\": 1}}.
    Observation: the result of the action
    ... (this Thought/ Action/ Action Input/ Observation  can repeat N times)

    ---
    **IMPORTANT RULE: As soon as you have enough information to definitively answer the user's original question, you MUST stop using tools. Your very last output must strictly follow the format below:**
    ---

    Thought: I now have the final answer and will provide it.
    Final Answer: [your final, complete, and well-formatted answer to the original question here]

    Begin!
    '''),
        HumanMessagePromptTemplate.from_template("{input}\n{agent_scratchpad}")
    ])

    # tools = [JupyterAPITool(), TATool(), TVLATool(), SPATool(), RAG_retrieve_and_summarize_Tool(retriever_dict=retriever_dict)]      # TraceCaptureTool(),
    # tools = [JupyterAPITool(), TATool(), TVLATool(), SPATool(), TraceCaptureTool(), RAG_retrieve_and_summarize_Tool(retriever_dict=retriever_dict)]
    tools = [JupyterAPITool(), TATool(), TVLATool(), TraceCaptureTool()]

    # Create the ReAct agent
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=prompt,
    )

    # Thought → (Action → Action Input → Observation)* → Thought → Final Answer

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=long_term_memory,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        return_intermediate_steps=True,
        max_iterations=40,  # 最多循环40步
        early_stopping_method="generate",
        # max_execution_time=60,     # 最多运行60秒
        # output_parser=ReActSingleInputOutputParser(),  # 支持 Final Answer 判断
    )

    return agent_executor, llm



if __name__ == "__main__":

    query = "Help me detect that the AES cryptographic algorithm implemented by a STM32F103 conforms to ISO/IEC 17825, and its operation clock frequency is 4800Hz. The TA trace with a 5M sampling rate is: ./trace_data/AES_STM32F103/AES_STM32F103_TA_fixed&random_plain_5M_1000traces_24000points.npy, The TVLA trace with a 5M sampling rate is: ./trace_data/AES_STM32F103/AES_STM32F103_TVLA_fixed&random_plain_5M_40000traces_15000points_power.npy, The names of trace information files are also given. Help me check whether the implementation is safe by ISO/IEC 17825."

    # 生成 Prompt（作为 system_message）
    prompt_text = inspect_prompt(query, agent_executor, long_term_memory, prompt, tools)

    # 生成带时间戳的日志文件名
    # a. 定义想要插入的自定义标签
    #主要就是：波形集、模型、配置（1单纯大模型、2带专业工具、3带知识库、4带专业工具及知识库）
    custom_tag = "AES_STM32F103_DeepseekV31T"

    # b. 生成带时间戳的部分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # # c. 将所有部分拼接成最终的文件路径
    # log_file_path = f"log/agent_log_{custom_tag}_{timestamp}.md"

    run_dir = f"log/agent_log_{custom_tag}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "agent_log.md")

    # 执行 Agent 并生成 Markdown 日志
    result = run_agent_and_log_md(
        agent_executor,
        query,
        llm=llm,
        prompt_text=prompt_text,
        log_path=run_dir
    )






