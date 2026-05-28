 # MiniBatch-LLM — Task Charter

  ## What
  从零写一个 LLM 推理服务,实现
  continuous(iteration-level)batching,
  并用一个诚实的 benchmark 证明它相对 static batching
  的吞吐↔延迟权衡
  (可选再对 vLLM)。单模型。

  ## Why
  AI-infra 主线项目。信号是 iteration-level scheduling +
  KV-cache 管理
  + roofline 级别的 benchmark 分析 —— 不是"FastAPI
    包了个模型"。

  ## Definition of done(整个项目)
  在 7B 模型、真 GPU 上,变长负载下,给出
  static-vs-continuous(vLLM 可选)
  的 throughput / p50 / p95 / p99 / tokens-per-sec 曲线,且
  README 要
  **解释**曲线(decode 受显存带宽限、KV-cache
  容量卡并发),不是只贴图。

  ## Invariants(贯穿所有 phase)
  1. ModelRunner 藏在单一接口后;server.py / scheduler.py 不直接
    import transformers,
     保证 runner 可换。
  2. scheduler 能用 FAKE runner(无模型、无 GPU)做确定性单测。
  3. 每个响应都带 metrics:queue_wait_ms、generate_ms、e2e_ms。
  4. 正确性先于性能:手写 decode loop 没和 HF generate() 逐
    token 对上之前,
     不碰 continuous batching。
  5. benchmark 诚实:变长输出、median-of-N、写清方法学、解释
    ceiling。

  ## Phase map(每 phase:build → self-review → polish → 过
  gate;一 phase 一 commit)
  - P0  static-batching scaffold(CPU)。Gate:并发正确性 +
    scheduler 单测绿。
  - P1  手写 greedy decode loop 驱动 past_key_values(不用
    generate())。
        Gate:与 model.generate(do_sample=False) 逐 token 一致。
  - P2  continuous batching(EOS 即踢、按 step
    从队列补;prefill-vs-decode 分裂;
        ragged position)。Gate:变长负载下,吞吐压过 static 且
    p95 不更差。
  - P3  GPU benchmark + roofline writeup;vLLM
    参考线可选。Gate:曲线 + 文字"为什么"。

  ## Non-goals(明确不做,除非以后提升)
  自写 PagedAttention、speculative decoding、多卡、训练/微调、
  非 greedy sampling 打磨、鉴权/多租户、SSE streaming。

  ## Stack
  Python · FastAPI · asyncio · PyTorch + HuggingFace
  transformers · pytest。
  GPU later:Bender A40/A100 或租的 NVIDIA。
  模型:distilgpt2(CI) → Qwen2.5-0.5B / TinyLlama-1.1B(本地) →
  Qwen2.5-7B / Llama-3.1-8B bf16(benchmark)。