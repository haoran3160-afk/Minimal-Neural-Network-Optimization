# Minimal Neural Network Optimization

用9个参数学习XOR的独立NumPy实验：手写反向传播、GD、动量和Adam。先读 [中文学习说明](reports/LEARNING_GUIDE.pdf)，再打开 [experiment.ipynb](notebooks/experiment.ipynb) 逐步重跑。

## 目录

```text
src/minimal_neural_network/  模型、优化器、梯度检查和实验入口
tests/                      梯度、稳定性和结构性质测试
notebooks/                  可交互重跑的实验
reports/                    学习说明及可编辑Markdown
results/                    tables/、figures/、artifacts/
.local/                     本地临时文件与缓存，不进Git
```

## 运行

在仓库根目录使用Python 3.12，建议先建立虚拟环境：

```powershell
python -m pip install -e .
python -m minimal_neural_network.check_gradients
python -m pytest -q
python -m jupyterlab notebooks/experiment.ipynb
```

重建全部实验：`python -m minimal_neural_network.run_experiment`。依赖统一来自`requirements.txt`；可编辑安装让Notebook从子目录启动时也能导入代码。

## 阅读顺序

1. [model.py](src/minimal_neural_network/model.py)：九个参数、稳定BCE与手写梯度。
2. [check_gradients.py](src/minimal_neural_network/check_gradients.py)：独立中心差分核对。
3. [train.py](src/minimal_neural_network/train.py)：三种优化器的更新方程。
4. [实验Notebook](notebooks/experiment.ipynb)：训练、20种子比较及真实失败案例。

固定主例用Adam在194次更新后得到BCE 0.0199167，四个真值表样本全部正确。完整结果同时保留失败：20个配对起点下GD成功12次、动量12次、Adam10次。全零初始化、隐藏层饱和、过大步长的卡点见 [学习说明源文件](reports/LEARNING_GUIDE.md)。

这是一种教学上很小的网络，不声称所有架构下绝对最小。四点训练准确率不等于独立泛化能力，固定学习率的比较也不证明某种优化器普遍更好。

`python -m minimal_neural_network.build_documents`根据结果更新说明和Notebook源文件，之后应重新运行Notebook恢复输出。代码与解释由AI协助完成，供你亲自重推、重跑和继续研究；本仓库与预算课程项目独立。
