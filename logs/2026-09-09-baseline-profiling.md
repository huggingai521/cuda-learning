# 2026-09-09 ComfyUI Baseline Profiling

第一次用 Nsight Systems 分析自己每天在跑的工作负载。

## 采集方式

```bash
nsys profile --trace=cuda \
  --output=/home/dl/cuda-learning/logs/comfyui_baseline \
  --force-overwrite=true \
  --delay=40 --duration=60 \
  python main.py --disable-auto-launch
```

**踩坑**：第一次用 Ctrl+C 停止，但按了两次，第二次打断了报告写入过程，
只在 /tmp 留下半成品 .qdstrm。改用 `--duration` 限时自动停止后正常。
教训：看到 `Generating ...` 之后什么都别按。

## Kernel 耗时 Top 5

| 占比 | 累计耗时 | 调用次数 | 说明 |
|---|---|---|---|
| 68.8% | 35.4 s | 8,480 | CUTLASS BF16 GEMM（256×128 分块，融合 relu） |
| 12.2% | 6.3 s | 1,280 | FlashAttention 前向 |
| 2.6% | 1.4 s | 5,490 | LayerNorm |
| 2.5% | 1.3 s | 14,212 | 逐元素拷贝（direct_copy） |
| 2.0% | 1.0 s | 7,654 | BF16 拷贝 |

**81% 集中在 GEMM + Attention**，这是 diffusion 模型的计算骨架。

## Kernel 名字字段拆解

| 字段 | 含义 |
|---|---|
| `80` | 为 sm_80（Ampere）编译 |
| `tensorop` | 使用 Tensor Core |
| `bf16` | BFloat16 精度 |
| `s16816` | MMA 指令形状 m16n8k16 |
| `relu` | 算子融合，GEMM 后直接做 relu，省一次显存往返 |
| `256x128` | threadblock tile 大小 |
| `32x3` | K 方向分块 32，3 级流水线（预取隐藏访存延迟） |
| `tn` | A 转置、B 不转置 |
| `align8` | 8 元素对齐，可用宽向量指令 |

## 内存传输

| 方向 | 总量 | 次数 | 中位数 |
|---|---|---|---|
| Host→Device | **33,479 MB** | 2,619 | 0.006 MB |
| Device→Device | 318 MB | 688 | ~0 |
| Device→Host | 70 MB | 54 | ~0 |
| memset | 26 MB | 56 | ~0 |

H2D 占绝对主导，其余三项合计不到其 1.3%。

## 带宽对照

| 通道 | 带宽 | 差距 |
|---|---|---|
| 显存 ↔ GPU | 1,344 GB/s | — |
| 内存 ↔ 显存（PCIe） | ~50 GB/s | **27 倍** |

这是"能不搬就不搬、能留显存就留显存"的量化依据。

## 两个真实课题

### 课题一：GEMM 跑的是 Ampere 代码

所有 GEMM kernel 前缀均为 `cutlass_80`（sm_80 = Ampere），
FlashAttention 及 `sm80_xmma_fprop_implicit_gemm` 同样是 sm_80 路径。
本机是 Blackwell（sm_120）。

**推测**：当前通过 PTX JIT 兼容执行 Ampere 代码，
未使用第 5 代 Tensor Core 的新特性（FP8/FP4 加速、新异步拷贝机制等）。

**待查**：编译目标、PTX JIT 机制、CUTLASS 版本、PyTorch TORCH_CUDA_ARCH_LIST。

### 课题二：33.5 GB H2D 传输

48 GB 显存，单次出图却搬入 33.5 GB，远超单个模型体积
→ 同一批权重被反复搬运。

中位数 6 KB、最大 67 MB 的双峰分布，说明既有大块权重分片，
也有大量零碎小传输（小传输成本几乎全是启动开销，不是带宽）。

**推测**：ComfyUI 默认显存策略偏保守（为小显存用户设计），
在 text encoder / UNet / VAE 之间反复卸载重载。

**待验证**：`python main.py --highvram` 或 `--gpu-only` 能否显著降低 H2D 总量。

**时间成本估算**：33.5 GB ÷ 50 GB/s ≈ 0.67 s（PCIe 5.0）
或 ÷ 25 GB/s ≈ 1.34 s（PCIe 4.0），相对 kernel 总耗时约 51 s，占 1–3%。
不是主要瓶颈，但需确认传输与计算是否重叠。

## 看不懂清单（未来 12 周的教学大纲）

- [ ] `cutlass` 是什么，为什么 PyTorch 用它
- [ ] `s16816` MMA 指令形状的含义
- [ ] `tn` / `align8` 的作用
- [ ] `32x3` 三级流水线如何隐藏延迟
- [ ] `xmma` 是什么
- [ ] `implicit_gemm`（卷积转矩阵乘）
- [ ] `cub` 库（DeviceSelectSweep / DeviceReduce）
- [ ] `splitKreduce` 为什么需要
- [ ] elementwise kernel 的模板参数 `<(int)128, (int)4>` 是什么
- [ ] PTX JIT 与架构兼容机制

## 下一步

- [ ] 试 `--highvram`，重新 profile 对比 H2D 总量
- [ ] 补齐 hardware.md 中的 SM 数量、L2 大小、共享内存配额
- [ ] W2 开始 C 子集学习
