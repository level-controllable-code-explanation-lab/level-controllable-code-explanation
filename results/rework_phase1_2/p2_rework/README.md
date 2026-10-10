# P1/P2/P3 口径修正记录（2026-10 豆包全盘审核 → Claude独立复核 → 用户拍板）

本目录存放CAEE论文第二轮返工（豆包3个severe问题审核）定稿时生成最终数字用的脚本，
对应修改已写入 `CAEE_MainText_realdata.docx` 和 `CAEE_AppendixA.docx`。

## 文件

本目录下的脚本统一引用上级目录的 `../td_metric.py`（权威47词表TD实现，`td()`默认使用其中的
`CANONICAL_TERMS`，与论文Appendix A.1-A.3声明的词表/匹配规则完全一致），不单独保留副本。

- `recompute_td_significance_n50.py`：P2定稿脚本。对5个受影响模型
  (ctrl_baseline/specialist_beginner/specialist_intermediate/specialist_expert/multilevel)
  在"47词表 + 真n=50独立样本"口径下重算TD均值和三对层间Mann-Whitney U/p/r。
  - ctrl_baseline特殊处理：该模型450条记录实际是150个"预测"×3份逐字节完全相同的导出拷贝
    (exp字段无seed区分)，真实样本量一直是n=50，去重时按code取首次出现即可(非3-seed均值)。
  - 其余4个模型(specialist_*/multilevel)exp字段确认有真实的s42/s123/s456三个独立训练种子，
    去重方式为3-seed TD均值法(每个code的TD = 该code在3个seed下的TD均值)。
- `recompute_target_td_train750.py`：P3定稿脚本。用权威47词表对`data/train_750.json`
  (750条真实训练参考，250条/层)直接计算target TD，取代Appendix A1陈旧的target值。
- `language_degradation_analysis.py`：Section 4.7语言退化现象 + Table 5定稿脚本。
  统计9个模型×3层级(27-cell)的"输出零中文字符"退化率；对3-seed模型(specialist_*/multilevel)
  做code级拆解(all3-degraded/none-degraded/mixed)；并计算zero_shot[intermediate]退化code集合
  与specialist_expert[intermediate]全退化code集合的重合度。复现数字：specialist_expert[intermediate]
  92%(138/150, code级44/0/6)、specialist_beginner[expert] 90%(135/150, code级43/4/3)、
  specialist_intermediate[intermediate] 5.3%(8/150，判定为随机噪音，code级0/42/8)、
  zero_shot/few_shot/ctrl_baseline/multilevel系列见脚本输出；13/13重合用于支持
  Table 5和4.7节"退化模式非距离训练分布单调关系"的讨论。

## 结论对照表（旧值 → 新值，均已写入论文）

### P2 — Section 4.3/4.5/4.7、Table 2、Fig.3 caption、Appendix A.3
| 模型 | b-i / b-e / i-e r值（旧，作废） | b-i / b-e / i-e r值（新，定稿） |
|---|---|---|
| multilevel | 0.749 / 0.920 / 0.865（n=150伪独立） | **0.799 / 0.914 / 0.862**（真n=50，3-seed均值） |
| CTRL-PO | 0.402 / 0.86 / 0.803（旧TD词表） | **0.549 / 0.914 / 0.867**（47词表） |
| specialist_expert (b-i) | 0.072 | **0.032**（仍不显著，p≈0.75，方向结论不变） |
| specialist_beginner | 未在正文单独引用 | 0.275 / 0.838 / 0.633（已入脚本输出，供SWARM核对） |
| specialist_intermediate | 未在正文单独引用 | 0.666 / 0.898 / 0.840（已入脚本输出，供SWARM核对） |

TD均值本身在新旧口径下不变（词表/独立性只影响显著性检验，不影响均值），Table 1/2/3的TD数值列未改动。

### P1 — Section 4.1 (正文para 42, "P43")
specialist_intermediate@intermediate层的BERTScore显著性(论文称p=0.004)，在真n=50独立样本下按不同
去重口径复算：seed42 p=0.0589 / seed123 p=0.3556 / seed456 p=0.0264 / 3-seed均值 p=0.0088。
p值在0.0088~0.3556间随口径摆动，部分口径Bonferroni(0.0167)后不显著。正文已改为"方向一致但对聚合
口径敏感"，不再声称"both remain significant after Bonferroni correction"适用于两个层级比较。
expert层的比较(p<0.001)在所有口径下保持稳健，未受影响。

### P3 — Appendix A.4、Table A1、正文para 52 (limitations)
Appendix A1的target TD值 0.185/2.835/29.135 系陈旧数据：溯源到`recompute_all.py`的
`agg("ref_td")`实现，证实只是对`results_v3/<model>.json`里预存的旧`ref_td`字段取均值，
该字段是数据准备阶段用rework前的52词表(未排除单字/O(n残缺)算的，非本论文声明的47词表口径。
新target用`recompute_target_td_train750.py`直接对训练参考文本重算：
**beginner 0.4413 / intermediate 3.0446 / expert 34.0260**。
Table A1达成率同步重算：12.2% / 60.6% / 90.1%（原105.2%"超过target"的表述已改为
"低于target"，因为expert层新target比旧值高，multilevel的TD从"略超target"变为"未达target"）。

## 待SWARM终审核对的关键点
1. ctrl_baseline去重后的r值(0.549/0.914/0.867)与`results/rework_phase1_2/ctrl_baseline_dedup_stats.json`
   的U/z值做交叉验证（该json用旧词表，r数值不同，但U/z应可独立复现同一组beginner/intermediate/expert
   TD原始分布的秩序关系）。
2. P1的四个口径p值(0.0088/0.0264/0.0589/0.3556)建议SWARM technical agent重新独立跑一次
   `verify_bertscore_pairwise.py`的扩展版本以交叉确认，而非只信任本次报告的数字。
3. tie-correction实现需要用`Counter(v for v, _ in allv)`对pooled数值计数，不能对`(value, group)`
   元组计数——本轮返工过程中发现并修正了一次这个bug（独立验证脚本早期版本曾因此产出错误的0.470）。
