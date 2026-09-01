# StructField-Vision

StructField-Vision 1.0.0 用于从灰度或 RGB 图像估计经过严格验证的概率结构场。
稳定核心无需 checkpoint 或外部数据，即可输出方向概率、频率矩、相干性、尺度证据，
以及明确标记为 baseline heuristic 的结构不确定性。

```python
import torch
from structfield import AnalyticStructuralFieldEstimator

field = AnalyticStructuralFieldEstimator()(torch.rand(2, 3, 96, 128))
modes = field.topk_orientations(k=2)
```

`orientation_prob` 是唯一持久化的方向状态；所有方向 mode 都动态推导。角度为
`[0, π)` 内的轴向弧度，在显示图像坐标中逆时针为正；频率单位为 cycles/pixel，
尺度单位为 pixels。v1 正式支持 Linux x86_64 与 Python 3.11。

命令行会原子发布 `structfield-estimate-v1` 不可变 bundle，并记录完整
`orientation_probability.npy`、文件大小和 SHA-256。安全 resolver 会拒绝符号链接、
未声明文件、schema 不匹配和哈希损坏。

仓库不分发数据集、预训练权重、临床表格或历史 checkpoint；示例和 README 材料均由
公开代码合成。研究 adapters、corruption、equivariance、reliability、不确定性
baseline 和三个参考应用统一标记在 `structfield.experimental` 下。

详见 [文档入口](docs/index.md)、[0.3 到 1.0 迁移](docs/migration-0.3.md)、
[限制](docs/limitations.md)、[代码来源](docs/legal/CODE_PROVENANCE.md) 与
[许可证范围](docs/legal/LICENSE_SCOPE.md)。本地发布流程不会 push、上传 PyPI 或创建
托管 Release。
