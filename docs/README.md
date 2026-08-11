# 📚 Study Aids

Supplementary reference material for the **Transformer Learning Journey**. The notebooks
(`../notebooks/`) are the main course — these documents are the textbook margin notes you
keep open beside them.

| Document | Use it to… |
|---|---|
| [**roadmap.md**](roadmap.md) | See the whole arc across all three tracks, the dependency graph, and recommended paths by role (research / training / inference / applications). |
| [**study-guide.md**](study-guide.md) | Plan your path: prerequisites, recommended order, per-notebook objectives, and self-check questions to confirm you actually *got* it. |
| [**glossary.md**](glossary.md) | Look up any term in one line — embeddings, attention, RoPE, MLA, MoE, GRPO, PagedAttention, RAG… with a pointer to the notebook that teaches it. |
| [**cheatsheet.md**](cheatsheet.md) | Grab the formula or tensor shape fast: attention, normalization, MoE, GRPO, quantization, the roofline, parallelism, serving, RAG. |
| [**architecture.md**](architecture.md) | Systems reference: memory budgets, the parallelism selection table, serving decisions, cost formulas, MFU/MBU targets. |
| [**variants-atlas.md**](variants-atlas.md) | Transformers beyond text — vision, speech, generative, science, decision-making, retrieval, code — each as *tokens / objective / delta from an LM*. |
| [**references.md**](references.md) | Go to the source. Authoritative papers, blog posts, courses, and videos, organized by topic and mapped to each notebook. |
| [**references-zh.md**](references-zh.md) | 中文参考资料 — Chinese-language resources (李沐论文精读、李宏毅课程、动手学深度学习、苏剑林博客等), mapped to each notebook. |
| [**diagrams.md**](diagrams.md) | Mermaid flowcharts for every stage across all 26 notebooks, plus generated figures. |

## How to use these

1. Skim **study-guide.md** first to see the whole arc and pick where to start.
2. Work through the notebooks in order. When a term is unfamiliar, check **glossary.md**.
3. Keep **cheatsheet.md** open for the math and tensor shapes.
4. After each notebook, follow the matching section of **references.md** to read the
   original paper — that is where "intermediate" becomes "expert."

> All references are to primary sources (original papers, official blogs, established
> courses). arXiv IDs have been verified.
