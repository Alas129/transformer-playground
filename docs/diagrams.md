# 图解与流水线 (Diagrams & Pipelines)

A visual companion to the notebooks. The **Mermaid** diagrams render automatically on
GitHub; the **PNG** figures (in [`images/`](images/)) are generated from real math by
[`images/generate.py`](images/generate.py) — re-run `python3 docs/images/generate.py` to rebuild them.

> 中文说明穿插在每节里。流程图说明「数据怎么流」，PNG 展示「真实的数值长什么样」。

---

## 0. 学习路径总览 (The whole journey)

```mermaid
flowchart TD
    subgraph FROM_SCRATCH["从零搭一个 GPT (NB 01–07)"]
        A[01 演进史<br/>RNN → Transformer] --> B[02 词嵌入 + 位置编码]
        B --> C[03 自注意力]
        C --> D[04 多头注意力]
        D --> E[05 Transformer Block]
        E --> F[06 完整架构]
        F --> G[07 训练并生成文本]
    end
    subgraph COMPLETE["补全全貌 (NB 08–11)"]
        H[08 训练深入<br/>loss / 调度 / 微调]
        I[09 推理与解码<br/>采样 + KV cache]
        J[10 编码器 & seq2seq<br/>BERT / T5]
        K[11 现代架构<br/>RMSNorm / RoPE / GQA]
    end
    subgraph POST["后训练对齐 (NB 12–13)"]
        L[12 指令微调 + LoRA<br/>SFT]
        M[13 偏好对齐<br/>RLHF / DPO]
    end
    G --> H --> I --> J --> K --> L --> M
```

---

## 1. 为什么需要 Transformer (NB 01)

RNN 顺序处理、难并行、长距离遗忘；Transformer 让每个 token 一次性看到所有 token。

```mermaid
flowchart LR
    subgraph RNN["RNN — 顺序、慢、健忘"]
        t1[Token1] --> t2[Token2] --> t3[Token3] --> t4[TokenN]
    end
    subgraph TF["Transformer — 并行、全局可见"]
        s1((Token1)) <--> s2((Token2))
        s2 <--> s3((Token3))
        s1 <--> s3
        s1 <--> s4((TokenN))
        s2 <--> s4
        s3 <--> s4
    end
```

---

## 2. 文本 → 张量 (NB 02)

```mermaid
flowchart LR
    txt["原始文本<br/>'The cat sat'"] --> tok["分词器<br/>tokenizer"]
    tok --> ids["token id<br/>[3, 41, 270]"]
    ids --> emb["词嵌入查表<br/>(vocab × d_model)"]
    emb --> add(("➕"))
    pos["位置编码<br/>sin / cos"] --> add
    add --> out["输入张量<br/>(seq_len × d_model)"]
```

正弦位置编码的真实数值（每行一个位置，每列一个维度）：

![Sinusoidal positional encoding](images/positional_encoding.png)

---

## 3. 自注意力的计算流程 (NB 03)

```mermaid
flowchart TD
    X["输入 X<br/>(seq × d)"] --> Q["Q = X·Wq"]
    X --> K["K = X·Wk"]
    X --> V["V = X·Wv"]
    Q --> S["scores = Q·Kᵀ / √dₖ"]
    K --> S
    S --> M["+ 因果掩码<br/>(屏蔽未来 token)"]
    M --> SM["softmax<br/>(每行和=1)"]
    SM --> O["输出 = 权重 · V"]
    V --> O
```

在一个玩具句子上算出的**真实**因果注意力权重——下三角，且每行归一化到 1：

![Causal self-attention weights](images/attention_weights.png)

---

## 4. 多头注意力 (NB 04)

```mermaid
flowchart LR
    X[输入] --> H1[head 1]
    X --> H2[head 2]
    X --> H3[head ...]
    X --> H4[head h]
    H1 --> C[拼接 concat]
    H2 --> C
    H3 --> C
    H4 --> C
    C --> P[线性投影 Wo]
    P --> Y[输出]
```

每个头在不同子空间里学不同的关系（语法、指代、位置…），拼接后再投影回 `d_model`。

---

## 5. Transformer Block (NB 05)

```mermaid
flowchart TD
    in[输入] --> ln1[LayerNorm]
    ln1 --> mha[多头自注意力]
    mha --> r1(("➕ 残差"))
    in --> r1
    r1 --> ln2[LayerNorm]
    ln2 --> ff["前馈网络<br/>(Linear → GELU → Linear)"]
    ff --> r2(("➕ 残差"))
    r1 --> r2
    r2 --> out[输出]
```

残差连接 + 归一化让深层网络可训练；这个 block 堆叠 N 次就是模型主体。

---

## 6. 完整 Decoder-only GPT (NB 06)

```mermaid
flowchart TD
    tok[token ids] --> emb[词嵌入 + 位置编码]
    emb --> b1[Transformer Block × 1]
    b1 --> b2[Transformer Block × 2]
    b2 --> bn[... × N]
    bn --> lnf[最终 LayerNorm]
    lnf --> head["输出头<br/>(d_model → vocab)"]
    head --> logits[logits]
    logits --> sm["softmax → 下一个 token 概率"]
```

---

## 7. 训练流水线 (NB 07–08)

```mermaid
flowchart LR
    data[语料] --> batch[切分 batch + 滑窗]
    batch --> fwd[前向: 预测下一个 token]
    fwd --> loss["交叉熵 loss<br/>(perplexity = exp(loss))"]
    loss --> bwd[反向传播]
    bwd --> opt["优化器 AdamW<br/>+ LR 调度 + 梯度裁剪"]
    opt --> fwd
    opt -. 周期性 .-> val[验证集评估]
```

训练用的**学习率调度**（线性 warmup + 余弦衰减，精确公式）：

![LR warmup + cosine decay](images/lr_schedule.png)

---

## 8. 推理与解码 (NB 09)

```mermaid
flowchart TD
    p[prompt] --> f[前向得到 logits]
    f --> dec{解码策略}
    dec -->|greedy| g[取 argmax]
    dec -->|top-k| tk[只在前 k 个里采样]
    dec -->|top-p| tp[nucleus: 累积概率 p]
    dec -->|beam| bm[束搜索保留 b 条路径]
    g --> nxt[追加 token]
    tk --> nxt
    tp --> nxt
    bm --> nxt
    nxt -->|未结束| f
    nxt -->|EOS / 达到长度| done[输出文本]
```

**KV cache**：自回归生成时缓存历史 token 的 K/V，每步只算新 token，避免重复计算。

---

## 9. 架构家族 (NB 10)

```mermaid
flowchart LR
    subgraph DEC["Decoder-only (GPT)"]
        d[因果注意力<br/>生成]
    end
    subgraph ENC["Encoder-only (BERT)"]
        e[双向注意力<br/>理解 / MLM]
    end
    subgraph S2S["Encoder-Decoder (T5)"]
        en[编码器] --> cross[交叉注意力] --> de[解码器]
    end
```

---

## 10. 现代 LLM 组件 (NB 11)

```mermaid
flowchart TD
    subgraph V2017["2017 原版"]
        a1[LayerNorm] --- a2[绝对位置编码] --- a3[MHA] --- a4[FFN + ReLU]
    end
    subgraph MODERN["现代 (LLaMA 风格)"]
        b1[RMSNorm] --- b2[RoPE 旋转位置编码] --- b3[GQA 分组查询注意力] --- b4[SwiGLU]
    end
    a1 -.升级.-> b1
    a2 -.升级.-> b2
    a3 -.升级.-> b3
    a4 -.升级.-> b4
```

外加 **FlashAttention**：不改数学结果，只优化显存读写，让长序列训练更快更省。

---

## 11. 后训练：从 base 模型到对齐助手 (NB 12–13)

```mermaid
flowchart LR
    base["预训练 base 模型<br/>(只会续写)"] --> sft["SFT 指令微调<br/>(指令-回答对, 可用 LoRA)"]
    sft --> rm["训练奖励模型<br/>(人类偏好排序)"]
    rm --> rlhf["RLHF (PPO)<br/>用奖励信号优化策略"]
    sft --> dpo["DPO<br/>直接用偏好对优化, 无需 RL"]
    rlhf --> aligned["对齐的助手模型"]
    dpo --> aligned
```

- **SFT**：教模型「按指令回答」，配合 **LoRA** 只训练少量低秩参数，省显存。
- **RLHF**：训练奖励模型 → 用 PPO 强化学习对齐人类偏好（InstructGPT 路线）。
- **DPO**：跳过奖励模型和 RL，直接用「优于/劣于」的偏好对做对比损失，更简单稳定。

---

> 想改图？Mermaid 直接编辑本文件即可；PNG 改 [`images/generate.py`](images/generate.py) 后重跑。
> 配套文字解释见各 notebook 与 [`study-guide.md`](study-guide.md)。
