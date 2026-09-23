---
trigger: always_on
---

<Simulating_Rules>
以下Rules适用于程序化虚拟仿真相关功能（"sweeper400\sweeper400\sim"目录下）的开发。本子包的核心是使用mph库通过 COMSOL Multiphysics 软件进行仿真计算研究。

### 1. COMSOL 仿真工作流

**标准流程**：
1. 用户首先在 COMSOL GUI 中创建和调试模型，保存到 `sim/mphs`（包含多个版本，除mph外均可作为文本文件读取，方便你查看仿真文件细节）
2. 使用 MPh 库连接 COMSOL Server（文档：https://mph.readthedocs.io/en/stable/）
3. 提取数据和可视化结果保存到 `storage/sim`

**MPh 使用要点**：
- 使用 `mph.start(cores=1)` 连接 COMSOL Server

### 2. 数据和可视化

- **数据存储**: 保存到 `storage/sim/[仿真名称+时间]/data/`；对于大矩阵式的参数扫描结果，可使用 `numpy.save()` 保存为 `.npy` 格式
- **图表输出**: 保存到 `storage/sim/[仿真名称+时间]/plots/`
- **Matplotlib中文显示**: 如需使用matplotlib时，建议在脚本开头使用项目config以优化中文字符显示：
```python
# 导入项目通用的matplotlib配置，并应用
from config.matplotlib_config import setup_chinese_fonts
setup_chinese_fonts()
```

## 典型工作流示例

### COMSOL 仿真自动化（详情可参考 `src/sweeper400/sim/参考代码1&2.py`）

1. 连接 COMSOL Server
2. 加载 `.mph` 模型
3. 运行仿真（参数扫描等）
4. 使用 `model.evaluate()` 提取数据
5. 保存数据到 `storage/sim`
6. 可视化分析

---

</Simulating_Rules>
