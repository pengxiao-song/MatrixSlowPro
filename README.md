计算图执行流程：

1. 初始化上游节点
2. 对结果节点调用 forward 方法（loss.forward）
3. 在变量节点上调用 backward 方法 (w.backward)
4. 做梯度下降（Optimizer 部分）
5. 清除所有节点的 value（前向传播） 和 jacobi（反向传播） 值，回到第（2）步
