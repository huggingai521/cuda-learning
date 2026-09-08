# CUDA 与推理优化学习记录

从零开始学习 GPU 编程与推理优化的过程记录。
起点：Python/SQL 背景，C++ 需重拾，无 CUDA 经验。
开始时间：2026-09-08

## 硬件

- NVIDIA RTX PRO 5000 Blackwell, 48GB GDDR7, sm_120
- NVIDIA RTX 3060, 12GB GDDR6, sm_86

详见 `notes/hardware.md`

## 目录

- `logs/` — 按日期的学习记录
- `experiments/` — kernel 代码与实验
- `notes/` — 知识笔记与硬件参数

## 进度

- [x] W1 环境搭建
- [ ] W1 ComfyUI baseline profiling
- [ ] W2-3 C 子集
- [ ] W4 第一个 kernel
- [ ] W5 内存层级
- [ ] W6 Reduction
- [ ] W7-8 矩阵乘
- [ ] W9-10 Transformer 算子
- [ ] W11-12 llm.c 与理论极限分析
