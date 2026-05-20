# 多模态客服智能体

DataFountain #1165 "睿创杯"第二届高校创新创业大赛 - 赛道三：具有多模态能力的客服智能体设计

## 技术方案

基于 **RAG（检索增强生成）** 架构，融合产品说明书知识库与通用客服规则，支持中英文问答和图片关联。

| 模块 | 技术 |
|------|------|
| Web框架 | FastAPI |
| Embedding | BAAI/bge-large-zh-v1.5 |
| 向量检索 | FAISS IndexFlatIP |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | MiMo-v2-Pro / MiMo-v2.5 |

## 成绩

| 指标 | 数据 |
|------|------|
| 总题数 | 400 |
| 成功率 | 100% |
| 含图片ID | 55.5% |
| 平均答案长度 | 522字符 |

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API密钥
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

### 3. 构建向量索引
```bash
python -m scripts.build_index
```

### 4. 启动API服务
```bash
python -m src.main
```

### 5. 测试
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk_customer_20260518" \
  -d '{"question": "DCB107指示灯闪烁是什么意思？"}'
```

### 6. 批量答题生成提交文件
```bash
python -m scripts.batch_answer --input question_public.csv --output submission.csv
```

## 项目结构

```
customer_service/
├── src/
│   ├── api/              # FastAPI 路由 (POST /api/v1/chat)
│   ├── core/             # ChatEngine, LLMClient, Prompts
│   ├── rag/              # ManualParser, Embedder, VectorStore, Reranker
│   ├── config.py         # pydantic-settings 配置
│   ├── app.py            # FastAPI 应用入口
│   └── main.py           # uvicorn 启动
├── scripts/
│   ├── build_index.py    # 构建向量索引
│   ├── batch_answer.py   # 批量答题
│   ├── fix_empty.py      # 修复空答案
│   └── analyze_dataset.py
├── docs/
│   ├── 技术方案文档.md    # 技术方案
│   ├── 验证报告.md        # 测试验证
│   └── ARCHITECTURE.md   # 架构设计
├── 手册/                  # 21份产品说明书 + 2608张插图
├── data/
│   ├── vector_store/     # FAISS索引
│   └── chunks/           # 文本块元数据
├── submission_fixed.csv  # 最终提交文件 (400题)
├── question_public.csv   # 公开题目
├── requirements.txt
└── .env.example
```
