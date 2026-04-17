# LLM 康复方案推荐系统 | LLM Rehab Recommendation System

> 一个基于大语言模型（LLM）的多模态医疗康复推荐系统，支持从患者病历文本、医学影像到最终康复方案的端到端智能生成。

---

## 项目简介

本项目是一个基于 **FastAPI** 的医学 AI 系统，旨在接收患者的多模态数据（结构化/非结构化文本、医学影像、交互式补充信息），自动完成 **疾病诊断 → 药物与治疗方案生成 → 排班整合**，最终输出完整的康复推荐方案。

目前 **Phase 1（多模态预处理）** 和 **Phase 2（智能疾病诊断）** 已经实现并可用，Phase 3–5 为预留扩展模块。

---

## 核心功能

- **多模态数据预处理**：同时处理文本病历和医学影像（CT / MRI / X-ray 等）。
  - **文本**：调用 DeepSeek LLM 提取医学实体、关系与摘要（`external/text_nlp.py`）。
  - **影像**：调用本地部署的 **HuatuoGPT-Vision** 模型进行病灶识别与影像描述（`external/image_analyzer.py`）。
- **知识增强的智能诊断**：基于 **ChromaDB** 向量知识库检索相关医学文献，结合 LLM 推理生成带置信度的疾病诊断列表（`services/diagnosis_service.py`）。
- **交互式诊断澄清**：当信息不足时，系统可自动向用户（医生/患者）发起多轮澄清提问，收集缺失信息后重新诊断。
- **模块化架构**：采用清晰的 `API → Service → External` 分层设计，便于后续扩展药物推荐、治疗计划与排班功能。

---

## 系统架构

![系统流程图](flow%20chart.png)

### 目录结构说明

| 目录                  | 说明                                                              |
| --------------------- | ----------------------------------------------------------------- |
| `api/`              | FastAPI 路由（`routes.py`）与请求/响应 Schema（`schemas.py`） |
| `services/`         | 核心业务逻辑                                                      |
| `external/`         | 外部模型（LLM / VLM）与数据库（ChromaDB）集成                     |
| `core/`             | 应用配置管理（`pydantic-settings`）                             |
| `models.py`         | Pydantic 领域模型                                                 |
| `utils.py`          | 日志配置与自定义异常                                              |
| `tests/`            | 端到端测试脚本                                                    |
| `huatuoGPT-Vision/` | 本地 HuatuoGPT-Vision 推理代码（**不含模型权重**）          |

---

## 快速开始

### 环境要求

- Python >= 3.10
- CUDA-capable GPU

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/llm-rehab-recommendation.git
cd llm-rehab-recommendation
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制环境变量模板并填入你的真实密钥：

```bash
cp .env.example .env
```

编辑 `.env` 文件，至少配置以下项：

```ini
# --- LLM 配置（SiliconFlow / DeepSeek）---
_API_KEY="你的 API Key"
BASE_URL="https://api.siliconflow.cn/v1"
MODEL_NAME="Qwen/Qwen2-7B-Instruct"

# --- ChromaDB 配置 ---
CHROMA_PERSIST_DIR_RELATIVE="external/chromaDB"
EMBEDDING_MODEL_NAME="BAAI/bge-large-zh"
CHROMA_COLLECTION_NAME="medical_markdown_data"
```

### 4. 准备本地模型与知识库

- **HuatuoGPT-Vision 模型权重**：将模型权重文件下载并放置到 `huatuoGPT-Vision/` 目录下。该目录已包含推理代码（`cli.py`、`llava/`），但**不包含大模型权重文件**，需自行准备。
- **ChromaDB 知识库**：系统依赖**预构建**的 ChromaDB 向量数据库。请确保 `external/chromaDB/` 目录下存在有效的 `chroma.sqlite3` 数据库文件及对应 collection。**项目不会自动**从 `external/kb_md_doc/` 中的 Markdown 文档构建数据库。

### 5. 启动服务

```bash
python main.py
```

服务将运行在 `http://127.0.0.1:8000`，默认开启热重载（reload）模式。你可以在浏览器中打开 `http://127.0.0.1:8000/docs` 查看自动生成的 Swagger API 文档。

### 6. 运行端到端测试

确保服务已启动后，执行：

```bash
python tests/run_e2e_test.py
```

该脚本会模拟一个完整的患者数据输入 → 预处理 → 诊断流程，并支持多轮交互式澄清的自动模拟。

---

## API 接口说明

### POST `/api/v1/preprocess` — 多模态预处理

接收患者原始数据，返回结构化的 `PreDiagnosisInfo`。

**请求示例：**

```json
{
  "patient_id": "P001",
  "text_data": [
    "患者，男性，68岁，长期吸烟史。",
    "主诉：持续性咳嗽，偶有痰中带血丝2月余。"
  ],
  "image_references": [
    "/absolute/path/to/lung_ct.png"
  ],
  "interactive_info": null
}
```

### POST `/api/v1/diagnose` — 智能诊断

接收 `PreDiagnosisInfo`，返回诊断结果或交互澄清请求。

**请求示例：**

```json
{
  "pre_diagnosis_info": { ... }
}
```

**响应示例（诊断完成）：**

```json
{
  "status": "Completed",
  "message": "Diagnosis process completed.",
  "diagnosis_result": {
    "request_id": "...",
    "diagnosis_list": [...],
    "primary_diagnosis": {...}
  }
}
```

**响应示例（需要交互）：**

```json
{
  "status": "Needs Interaction",
  "message": "Diagnosis requires further clarification.",
  "interaction_needed": {
    "questions_to_user": ["患者中风是缺血性还是出血性？"]
  }
}
```

---

## 开发说明

- **导入路径**：部分模块（如 `core/config.py`、`external/llm_client.py`）通过修改 `sys.path` 实现从项目根目录的导入。在移动或重命名文件时请注意保持导入路径的可用性。
- **HuatuoGPT-Vision**：采用懒加载（lazy initialization）+ 异步锁（async lock）机制，确保模型只在首次调用时加载一次，避免重复初始化。
- **诊断解析容错**：LLM 被提示返回严格 JSON，但解析器（`_parse_llm_diagnosis_response`）具备 JSON 提取失败后的关键词/启发式文本解析回退能力。

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

## 免责声明

⚠️ **本项目仅供学术研究、教学演示和编程学习使用，不构成任何医疗建议、诊断或治疗方案。** 任何基于本项目输出的医学决策均需谨慎对待，并应在持有执业资格的医务人员指导下进行。作者不对因使用本项目而产生的任何后果承担责任。

---

## English Abstract

**LLM Rehab Recommendation System** is a multimodal medical AI pipeline built on **FastAPI**. It ingests patient text records and medical images, leverages a local **HuatuoGPT-Vision** model for image analysis, a **DeepSeek** LLM for text understanding, and a **ChromaDB** vector knowledge base for retrieval-augmented diagnosis. Currently, **Phase 1 (Preprocessing)** and **Phase 2 (Diagnosis with interactive clarification)** are fully implemented, while Phases 3–5 (Recommendation, Scheduling, and Reporting) are reserved for future extension.
