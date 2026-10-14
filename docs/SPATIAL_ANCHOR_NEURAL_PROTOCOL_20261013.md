# 固定空间基底上的双原型残差：B 系列预登记

依据：v1.5方案已经提出固定alpha=0.5的空间基底；原七切片与扩充六来源的独立参考选参空间基线均出现小幅开发改善。此前A2复杂残差尚未超过Ridge。新的问题是神经残差能否解释简单空间基底之后的误差。固定0.5与已有参考选alpha的基线是不同候选，分别报告。

本轮先实现B2的两个匹配残差目标，保留state/niche Student与EMA Teacher、蛋白条件CSNN及原先全部损失/预算。唯一相对于匹配C0/CF0的架构变化是部署和训练基底均改为`b_sp = b_RNA + 0.5 * center(neighbor_mean(b_RNA)-b_RNA)`。对称k6、半径2.5、整片非孤立spot中心化；原始RNA anchor仍冻结。没有逐查询或逐蛋白调整alpha，alpha是冻结buffer，不进入优化器。B1 state-only配置用于后续有限消融，不同时增加多尺度niche或新增强。

训练内版本以允许参考的完整anchor产生b_sp；cross-fitted版本对每个训练来源g，用独立拟合于A-g的anchor产生g的RNA预测及其空间基底。该anchor的HVG、scaler、正则选择都排除g；同来源其他切片也排除。空间操作只能读目标切片RNA和坐标，不能读其ADT。辅助Student头仍监督真实Y。验证和部署都用完整允许参考anchor的空间基底。

最终`prediction = b_sp + gamma_state*delta_state + gamma_local*delta_local`。两个神经开关关闭时，精确回退b_sp；alpha=0且神经开关关闭时精确回退原RNA anchor。在相同FP32 anchor输入上，邻域求和/中心化使用FP64并转回FP32，与已完成的简单空间控制一致；CPU/GPU anchor矩阵乘法差异另以FP32容差记录。

warm-up8+ramp8、80轮上限、每4轮验证、patience6、seed2025、无mask/count thinning，与冻结C0/CF0匹配。门控规则仍是至少4参考来源、1%平均相对改善、60%来源改善。全部嵌套held来源必须排除于参数、anchor、特征、epoch选择、原型初始化；预测和评分分离。

代码在独立`stprotein2_spatial_anchor`命名空间，禁止修改正在运行的C0/CF0及竞争者源码。继承原表示与算子，保存独立源码快照、基底alpha、raw-RNA/base/spatial/state/local分解和完整训练目标来源。

验证：alpha0与原C0/CF0的预测/损失/全部参数梯度及真实训练更新一致；alpha0.5与独立NumPy基线一致；非零修正下分块预测和梯度一致；闭门回退；OOF空间目标来源排除；修改基底、图配置或输入阻止恢复；中断恢复；完整六来源合成嵌套31拟合与query ADT隔离。

真实初步诊断使用与旧C0/CF0完全相同的内层划分、16轮预算和anchor：参考Tonsil_AddOns/Tonsil/Breast，验证Glioblastoma。只比较该内层MSE与相同验证上的冻结空间基底；不能作为外层准确率或独立确认。先完成实现检查再运行。完整外层矩阵的启动须结合当前冻结C0/CF0的结果与该诊断；不预先宣称有收益。
