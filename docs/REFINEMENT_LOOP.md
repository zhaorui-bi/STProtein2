# 当前完整迭代循环

当前用户目标是循环完成研究收尾，确定最精炼、最合适的STProtein2版本，再在其上逐项增加novelty。

已完成的306项核心证据与40张完整开发网格位于研究目录`/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/stprotein2_core_20261025_v1/`。验证后的B0底座已保存为提交`2f5f573`和标签`baseline-b0-20261025`，保留可重放的基准，不直接宣布它是最终最佳模型。

新一轮精简候选确认只新增24项：自身RNA MLP与普通邻域MLP、两个种子、全部六来源；复用已有A40 seed2025的12项。状态与实际进程见[运行目录](/fs04/scratch2/nd32/zhaorui/STProtein/STProtein2/runs/stprotein2_compact_confirmation_20261025_v1/README.md)。旧339项暂缓矩阵不恢复。

干净实现位于`/fs04/scratch2/nd32/zhaorui/STProtein2-clean`的`compact-refinement-20261025`分支。可选残差组件已通过非零输出、梯度和AdamW更新检查，并完成两个真实4164×31切片预测重放；完整神经训练/导出/部署接口和最终默认版本仍待验收。[整体证据清单](/fs04/scratch2/nd32/zhaorui/STProtein2-clean/docs/REFINEMENT_LOOP.md)保留全部未完成项。

模型取舍须等完整三种子结果，报告严格最低均值及精简偏好两种选择、负例和波动。外部baseline与独立确认要求继续保留；预留外部蛋白不能用于当前版本或novelty的调参。新机制在精简版本固定后逐项加入，新增机制与预测收益分别验证。
