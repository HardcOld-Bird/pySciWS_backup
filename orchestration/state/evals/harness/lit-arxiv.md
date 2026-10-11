member: lit
tier: minimum
dirs: data/research/eval_scratch/lit-arxiv
max_turns: 12
timeout_s: 600
产物: data/research/eval_scratch/lit-arxiv/shortlist.md check=none
断言: 2310\.16699
断言于: data/research/eval_scratch/lit-arxiv/shortlist.md
禁止标签: whitelist_conflict
说明: E2 次案——最低成本档能否跑通**检索 → 结构化落盘 → 交付格式**。判据=指定
  arXiv ID 出现在产物文件里（断言面=文件，不是正文，防「嘴上说了、文件里没有」）。
  网络：默认直连，禁走代理。

## 任务书

用 arXiv 源检索这篇论文的准确标题，并把命中结果落盘成简版 shortlist：

    pysci-research search 'Exceptional points in non-Hermitian photonics: Applications and Recent Developments' --source arxiv --json

**落盘位置（本任务白名单，勿写到别处）**：
`data/research/eval_scratch/lit-arxiv/shortlist.md`

文件格式（delivery 要能被机械断言，别改标题行结构）：

    # arXiv shortlist — <检索词>

    - **Exceptional points in non-Hermitian photonics: Applications and Recent Developments**
      arXiv:2310.16699 · Meng et al. · 2023 · 相关性：一句话说明它为什么与本所相关

至少列出本次检索命中结果中 **前 5 条**（标题、arXiv ID、第一作者姓、年份各一行），
其中必须包含 arXiv ID `2310.16699`（这条是判据锚点，检索结果里本就有它）。若首次
检索的 15 条里没有它，换更精确的检索词重试，直到命中；不要手工编造 ID。

不要访问付费墙、不要下载全文、不要动 `data/skills/literature_research/` 正式知识库。
交付走 `<result>`，产物声明：

    <artifact check="none" reason="文献 shortlist 无注册机械检查，判据由 harness 的断言面执行">data/research/eval_scratch/lit-arxiv/shortlist.md</artifact>
