---
type: "always_apply"
---

<Basic_Rules>

Environments:
- We are working with Python (miniconda).
- Python interpreter version: 3.12.11
- Some installed packages: nidaqmx, scipy, matplotlib, numpy, pytest, pytest-cov, pydantic, ruff, pre-commit, commitizen
- If you need to install a new package, please stop responding and let me handle it manually. Do not install it by yourself.

Main Task:
- 我们正在开发 "sweeper400" package，它的主要功能是：协同控制NI数据采集卡（使用 "nidaqmx" package）和步进电机（使用"MT_API.dll"文件），自动化分步采集空间中不同位置的信号，并对信号进行处理，获取信号的空间分布等信息。
- "sweeper400" package 包含以下子包："measure"（包含NI数据采集相关module），"move"（包含步进电机控制相关module），"analyze"（包含信号和数据处理相关module），"use"（协同调用其他子包，将功能封装为适用于特定任务的专用对象，供外部以简洁的方式使用），"gui"（使用flet创建GUI应用程序，方便用户使用）

Detailed Rules:
- 请在开发中遵循以下方式："sweeper400" package的所有文件位于根目录的"src/sweeper400"目录中，测试代码则位于根目录的"tests"目录中。请在"src/sweeper400"目录中编写各子包/模块源代码（实现所有的函数/类/方法/属性），在"tests"目录中编写测试代码调用"sweeper400" package（已使用开发模式安装，可以直接import）。
- 由于本项目涉及硬件交互，而硬件测试往往速度很慢（例如步进电机的运动或NI机箱的数据采集），因此暂不维护全覆盖的测试套件，仅对部分重点功能编写pytest测试。要求在"tests"目录中按照"src"的镜像结构组织测试代码，并适配pytest和pytest-cov。推荐使用pytest的测试类（Test Classes）组织测试代码。
- 本包配置了日志管理框架"sweeper400\sweeper400\logger.py"，推荐在开发中使用它统一进行日志管理，合理为我们的代码配置日志输出，方便我们监测代码的运行情况。
- 开发过程中，请不要忘记适时更新各级"__init__.py"和package配置文件"sweeper400\pyproject.toml"。
- 你暂时不需要主动创建“使用示例/演示脚本”和“使用指南/说明文档”，只要将Docstring和代码注释写得清楚详细即可。
- "scripts"目录中是一些手工编写的脚本吗，用于供我手动检测和使用该包的功能。请你忽略该目录，不要修改或运行这些文件——它与我们目前的工作无关。

</Basic_Rules>
