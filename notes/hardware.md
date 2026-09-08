# 硬件参数

## Roofline 关键数字

| 参数 | 值 |
|---|---|
| 显存带宽 | 1,344 GB/s |
| BF16/FP16 稠密算力（估算） | ~258 TFLOPS |
| **算术强度拐点** | **~192 FLOP/Byte** |

拐点含义：每读 1 字节若做不到 192 次浮点运算，即为 memory-bound。
注：算力从官网 "AI TOPS 2064"（FP4 稀疏口径）逐档减半推算，非精确值。
待第 7-8 周矩阵乘实测后替换。

## RTX PRO 5000 Blackwell

- 架构：Blackwell, sm_120
- 显存：48GB GDDR7 with ECC
- 带宽：1,344 GB/s
- Tensor Core：第 5 代
- 功耗：300W
- SM 数量：<待 deviceQuery 填>
- L2 缓存：<待填>
- 每 SM 共享内存：<待填>

## RTX 3060

- 架构：Ampere, sm_86
- 显存：12GB GDDR6
- 带宽：约 360 GB/s
- SM 数量：<待填>

## 软件栈

- Ubuntu 24.04.4 LTS
- 驱动：580.173.02（open kernel module）
- CUDA Toolkit：12.8.93（同时保留 11.7 作回退）
- Nsight Systems 2024.6.2 / Nsight Compute 2025.1.1
- PyTorch 2.10.0+cu128（ComfyUI venv: /home/dl/comfy-env）

## 编译参数模板

nvcc -gencode arch=compute_86,code=sm_86
-gencode arch=compute_120,code=sm_120
-gencode arch=compute_120,code=compute_120
xxx.cu -o xxx
