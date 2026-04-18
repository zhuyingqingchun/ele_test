# ele_servo_new

## 项目路径

```
/mnt/PRO6000_disk/swd/ele_servo_new/
```

## 目录结构

```
ele_servo_new/
├── .gitignore                                    # Git 忽略配置
├── build_paper_repro_bundle.sh                   # 构建论文复现包脚本
├── run_all_paper_pipeline.sh                     # 总控流水线脚本
├── paper_repro_bundle_README.md                  # 论文复现包说明
├── paper_repro_bundle_manifest.txt               # 论文复现包清单
├── paper_repro_bundle_requirements.txt           # 论文复现包依赖
│
├── ele_servo_paper_repro_bundle/                 # 论文复现包
│   ├── src/                                      # 数据集构建与流水线脚本
│   ├── servo_diagnostic/                         # 伺服诊断模块
│   ├── servo_llm_alignment/                      # 伺服-LLM 对齐模块
│   ├── experiments_smoke_20260316/               # 实验脚本
│   ├── 论文/                                     # 论文正文与图表
│   ├── 项目拓扑图/                               # 项目拓扑图
│   ├── model/                                    # 模型定义
│   ├── handoff_packages/                         # 交接包
│   └── *.png / *.md                              # 说明文档与图示
│
├── signal_prefix_qwen_reasoning_rag_20260402/    # RAG 推理项目
│   ├── code/                                     # 源码
│   │   ├── servo_diagnostic/                     # 伺服诊断模块
│   │   ├── servo_llm_alignment/                  # 伺服-LLM 对齐模块
│   │   ├── experiments_smoke_20260316/           # 实验脚本
│   │   └── src/                                  # 数据集构建脚本
│   └── docs/                                     # 文档
│       └── README.md
│
└── count_files.sh                                # 目录统计脚本
```

## 已忽略的目录与文件

- `raw_data/`, `derived_data/`, `derived_datasets/`, `diagnostic_datasets/`
- `models/`, `checkpoints/`, `results/`
- `comparison_plots/`, `traditional/`
- `*.npz`, `*.npy`, `*.pkl`, `*.pt`, `*.pth`, `*.jsonl`
- `logs/`, `*.log`, `caches/`, `venv/`

## 远程仓库

- **URL**: `https://github.com/zhuyingqingchun/ele_test.git`
- **分支**: `main`
