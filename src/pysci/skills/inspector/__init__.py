"""inspector 技能：闲时巡检组员的专属工具集（``pysci-inspect``）。

与 ``pysci-orch``（组长）、``pysci-dev``（devops）、``pysci-bench``（bencher）严格四分
（REFACTOR v1.0 §0.9/§7）：本包**只做发现登记与分流**——把巡检看到的写成一条
``findings.jsonl`` 记录，判断类经共享仲裁库出判决，机械类免判决晋升。

生命周期与资源闸门（开关/预算/让路/迟滞）不在这里，在
:mod:`pysci.skills.orchestration.tools.inspectd`（组长侧 ``orch inspector-*``）——
组员不该、也没有能力决定「什么时候允许自己花资源」。**本 CLI 没有 override 参数**：
§5.2 的「inspector 无自裁面」靠的是代码里不提供这个选项，而不是文档里劝一句。
"""
