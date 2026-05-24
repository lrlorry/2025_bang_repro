# BANG Base 各规模资源计划

> 更新日期：2026-05-24。  
> 估算工具：`scripts/estimate_resources.py`（运行 `python3 scripts/estimate_resources.py` 获取最新数字）。  
> 硬件参考：论文使用 A100 80GB + 640 GB host RAM（SIFT1B/DEEP1B 场景）。

---

## 资源估算公式

### Host RAM（graph index）
```
每个 graph entry（bang_preprocess.py 处理后连续格式）：
  = dim × sizeof(T) + 4 + R × 4  （full vector + neighbor_count + neighbor_ids）

总大小 = N × entry_bytes

注：full vectors 嵌入在 graph entry 中，不单独占用额外 RAM
```

### GPU HBM（PQ compressed + per-query buffers）
```
PQ compressed vectors = N × uChunks bytes（uint8，uChunks ≈ B_gb×GiB/N）
PQ table（转置）= D × 256 × 4 bytes
per-query buffers（numQ=10000, L=100）主要项：
  d_pqDistTables  = numQ × uChunks × 256 × 4  （最大项）
  d_FPSetCoordsList = MAX_PARENTS × numQ × D × sizeof(T)  （rerank 缓冲）
  d_processed_bit_vec = numQ × 399887 bytes  （Bloom filter）
```

### 磁盘
```
原始 base vectors + graph index + PQ 文件（compressed + pivots）
通常 = base + graph + 少量 PQ meta
graph > base（因为包含 full vectors + adjacency list）
```

---

## 各规模详细计划

### SIFT10K（已有预构建文件）

| 项目 | 值 |
|---|---|
| N | 10,000 |
| dim | 128，float32（4B）|
| R | 64 |
| uChunks（估算）| ~125（1.2MB / 10K）|
| base vectors | ~5 MB |
| graph index（host）| ~7 MB（已有：sift10k_index_disk.bin 7.4 MB）|
| PQ compressed（GPU）| ~1.2 MB（已有）|
| GPU HBM 总计 | < 1 GB（A100 80GB 绰绰有余）|
| host RAM 总计 | < 100 MB |
| 磁盘 | ~10 MB（已有）|
| 数据来源 | 已有预构建文件（sift10kfiles/）|
| 构图耗时 | 不需要（已有）|
| **风险** | 预编译二进制可能 CUDA driver 不兼容 |

**下一步**：`bash scripts/run_sift10k_smoke.sh`（需 GPU）

---

### SIFT1M（base 已有，index 未构建）

| 项目 | 值 |
|---|---|
| N | 1,000,000 |
| dim | 128，float32（4B）|
| R | 64 |
| B_gb | 1.0（DiskANN `-B 1`）|
| uChunks | ~107（1GiB/1M ≈ 1073 B/vec，但 DiskANN 会对齐/取整）|
| base vectors（已有）| 488 MB（sift1m_data/sift1m_base.bin）|
| graph index（host）| ~700 MB（N × (128×4 + 4 + 64×4) = ~700 MB）|
| PQ compressed（GPU）| ~107 MB（N × 107 bytes）|
| per-query GPU buf（numQ=10000）| ~2.5 GB（d_pqDistTables = 10K × 107 × 256 × 4 ≈ 1.1 GB）|
| GPU HBM 总计 | ~2 GB（A100 足够）|
| host RAM 总计 | ~1.5 GB（graph）|
| 磁盘 | ~2 GB（base + graph + PQ）|
| 数据来源 | base/query/gt 已有；index 需 DiskANN 构建 |
| 构图耗时 | ~5 分钟（1M, float32, A100）|
| **风险** | LOAD-01：float 路径（BANGSearchInner<int> 碰巧正确，float=4B=int）|

**前置步骤**：
1. 安装编译 DiskANN
2. `build_disk_index --data_type float --dist_fn l2 --data_path sift1m_base.bin --index_path_prefix sift1m_index -R 64 -L 200 -B 1 -M 48`
3. `python bang_preprocess.py sift1m_index_disk.index sift1m_index_disk.bin 128 1 64`
4. `bash scripts/build_bang_base.sh`
5. `./bang_search sift1m_index sift1m_query.bin sift1m_groundtruth.bin 10000 10 float l2`

---

### SIFT10M（需下载 + 构图）

| 项目 | 值 |
|---|---|
| N | 10,000,000 |
| dim | 128，uint8（1B）|
| R | 64 |
| B_gb | 1.0（DiskANN `-B 1`）|
| uChunks | ~107 |
| base vectors | ~1.3 GB（uint8）|
| graph index（host）| ~8 GB（N × (128 + 4 + 64×4) = ~8.6 GB）|
| PQ compressed（GPU）| ~1 GB（N × 107 bytes）|
| per-query GPU buf（numQ=10000）| ~2.5 GB |
| GPU HBM 总计 | ~3.5 GB（A100 足够）|
| host RAM 总计 | ~10 GB（需 ≥ 16 GB RAM）|
| 磁盘 | ~12 GB |
| 数据来源 | 从 SIFT-1B 截取前 10M（bigann）或 big-ann-benchmarks |
| 构图耗时 | ~30 分钟 |
| **风险** | LOAD-01：uint8 路径存在类型安全问题，需先验证 |
| **AutoDL 注意** | 数据下载需要访问法国服务器（irisa.fr），速度可能慢；总磁盘约 12 GB，注意数据盘空间 |

**数据获取**：
```bash
# 方法 1：big-ann-benchmarks（推荐）
python create_dataset.py --dataset bigann-10M

# 方法 2：从 SIFT-1B 截取（需先下载 ~128 GB）
# 不推荐
```

---

### SIFT100M / DEEP100M（资源需求大）

| 项目 | SIFT100M | DEEP100M |
|---|---|---|
| N | 100,000,000 | 100,000,000 |
| dim | 128（uint8）| 96（float32）|
| base vectors | ~13 GB | ~36 GB |
| graph index（host）| ~86 GB | ~100 GB |
| PQ compressed（GPU）| ~10 GB | ~10 GB |
| GPU HBM 总计 | ~12 GB | ~12 GB |
| host RAM 总计 | ~90 GB | ~110 GB |
| 磁盘 | ~110 GB | ~150 GB |
| 构图耗时 | ~2-4 小时 | ~3-5 小时 |
| **风险** | 需要 ≥ 128 GB RAM 机器 | 需要 ≥ 128 GB RAM 机器 |
| **AutoDL 推荐实例** | 128 GB+ RAM 配置 | 128 GB+ RAM 配置 |

**DEEP100M 特殊说明**（来自 `BANG-Billion-Scale-ANN/README.md`）：
> DEEP100M is to be cut out from DEEP1B. Take first 100M points. https://big-ann-benchmarks.com/

---

### SIFT1B / DEEP1B（论文规模）

| 项目 | SIFT1B | DEEP1B |
|---|---|---|
| N | 1,000,000,000 | 1,000,000,000 |
| dim | 128（uint8）| 96（float32）|
| base vectors | ~128 GB | ~360 GB |
| graph index（host）| ~640 GB（论文 README）| ~640 GB（论文 README）|
| PQ compressed（GPU）| ~100 GB（A100 80GB 需压缩）| ~100 GB |
| GPU HBM | A100 80GB | A100 80GB |
| host RAM | ≥ 640 GB | ≥ 640 GB |
| 磁盘 | ≥ 800 GB | ≥ 800 GB |
| 构图耗时 | 数小时~ 数十小时 | 数小时~ 数十小时 |
| **风险** | 极高：需服务器级 RAM；uint8 LOAD-01 问题；PCIe 带宽关键 | 同左 |
| **AutoDL** | 不建议在 AutoDL 上尝试（内存/磁盘不足）| 同左 |

**PQ 压缩策略**（SIFT1B）：
```
A100 80GB，需要 GPU 存下 PQ compressed vectors
N=1B，uChunks=80 → 1B × 80 = 80 GB（刚好放入 80GB A100）
DiskANN 参数：-B 80（80 GiB）
```

---

## 复现优先级建议

```
优先级 1（本机无 GPU 可验证流程）：
  └─ SIFT10K smoke test（已有文件，命令已知）

优先级 2（AutoDL A100 小规模复现）：
  └─ SIFT1M
       ├─ base/query/gt 已有（sift1m_data/）
       ├─ 需要 DiskANN 构图（~5 分钟）
       └─ 可复现 recall vs QPS 曲线

优先级 3（AutoDL A100 中规模复现）：
  └─ SIFT10M
       ├─ 需下载数据（~1.3 GB base）
       ├─ 需要验证 uint8 路径（LOAD-01）
       └─ 需要 16 GB+ RAM

优先级 4（论文规模，需服务器）：
  └─ SIFT100M / DEEP100M / SIFT1B / DEEP1B
```

---

## AutoDL 实例选择建议

| 复现规模 | GPU | host RAM | 存储 | 建议实例 |
|---|---|---|---|---|
| SIFT10K | A100 40GB+ | 4 GB | 系统盘 50GB 足够 | 任意 A100 实例 |
| SIFT1M | A100 40GB+ | 8 GB | 2 GB 数据盘 | 任意 A100 实例 |
| SIFT10M | A100 40GB+ | 16 GB | 20 GB 数据盘 | A100 + 数据盘扩容 |
| SIFT100M | A100 80GB | 128 GB | 150 GB 数据盘 | 高内存配置 |
| SIFT1B | A100 80GB | 640 GB+ | 1 TB+ | 不建议 AutoDL |

**关键提醒**：
- `/root/autodl-tmp/` 是数据盘，下载数据和 index 文件必须放在这里
- `numCPUthreads=64` 硬编码，AutoDL CPU 核数可能不是 64（用 `nproc` 确认，如不一致需改源码重编）
