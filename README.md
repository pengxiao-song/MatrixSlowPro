TODO:

1. 规范标准机器学习流程：训练集和测试集
2. 提供数据集借口
3. 重构 example 函数
4. 提供 test 文件

Update：

1. 提供数据集接口 matrixslowvision 以及 dataloader 等（参考pytorch）
2. 重构代码，微调组织架构（更清晰），并提供完整项目架构图
3. 规范流程
4. 修复 bug

计算图执行流程：

1. 初始化上游节点
2. 对结果节点调用 forward 方法（loss.forward）
3. 在变量节点上调用 backward 方法 (w.backward)
4. 做梯度下降（Optimizer 部分，具体来说就是怎么更新）
5. 清除所有节点的 value（前向传播） 和 jacobi（反向传播） 值，回到第（2）步
