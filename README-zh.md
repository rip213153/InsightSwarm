以下是该文档的中文翻译：

# InsightSwarm

InsightSwarm 是一个本地优先、基于共享存储的多智能体研究运行时（runtime），旨在将研究问题转化为带有引用的报告。

当前原型专注于通过基于 SQLite 的共享存储进行协作的自治工作节点（workers），而不是由中央控制器像调用函数一样调用智能体。一个轻量级的运行时负责引导启动、启动工作节点、监控预算和交付条件，并将输出写入本地项目目录。

## 当前架构

- **Researcher**（研究员）：负责搜索、抓取、评估来源质量、发布原始来源，并可以使用私有作用域下的子智能体进行并行来源发现。
- **BrowserAgent**（浏览器智能体）：通过可见的浏览器/CDP路径、代码驱动的页面检查以及高风险操作的人工授权，来处理复杂的网页数据获取。
- **Extractor**（提取器）：将原始文档转换为附带原文引用的正式证据。
- **Critic**（评审员）：审查特定范围内的证据包，质询薄弱的内容覆盖，并可以请求针对性的修正。
- **Writer**（撰写者）：在交付门控（delivery gate）开启后，创建最终的 `report`（报告）、`report_partial`（部分报告）或 `report_blocked`（受阻报告）。
- **Lead**（主导者）：引导初始工作并维护约束条件，但各工作节点通过共享的任务、消息、制品（artifact）和证据存储来自主推进流程。

## 安装

```powershell
git clone <repo-url>
cd InsightSwarm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[browser]"
```

将 `.env.example` 复制为 `.env`，或直接设置环境变量。
将 `config.models.example.json` 复制为 `config.models.json`，然后只修改
provider 的 `base_url`、`api_key_env` 和模型名即可切换 OpenAI-compatible 端点。

```powershell
$env:MODEL_API_KEY="..."
$env:TAVILY_API_KEY="..."
$env:INSIGHTSWARM_MODEL_CONFIG="config.models.json"
$env:INSIGHTSWARM_MODEL_PROVIDER="default"
```

可选配置：

```powershell
$env:FIRECRAWL_API_KEY="..."
```

切勿提交真实的 API 密钥。`.env` 文件、运行数据库、生成制品、浏览器配置文件以及临时输出均已被 `.gitignore` 忽略。

## 运行

在 Windows 终端中提问中文问题时，请使用 UTF-8 编码：

```powershell
chcp 65001
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

运行一个实际的研究问题：

```powershell
python -m insightswarm.cli --model-provider default run ask "为什么2026年中国航空公司燃油费屡次上调" --search-provider tavily --max-runtime-seconds 1800 --max-no-progress-seconds 180 --max-drain-seconds 900
```

在复杂的网页获取中使用可见的浏览器路径：

```powershell
$env:INSIGHTSWARM_BROWSER_BACKEND="visible"
$env:INSIGHTSWARM_BROWSER_PROFILE_ROOT="E:\code\InsightSwarm\.tmp\browser-profiles"

python -m insightswarm.cli --model-provider default run ask "了解这个复杂网站" --browser-backend visible --search-provider tavily
```

附加本地多模态输入（如图片）作为用户提供的上下文：

```powershell
python -m insightswarm.cli --model-provider default run ask "我想了解这张图片里的网站" --input-file "C:\path\to\image.png" --browser-backend visible
```

输出结果将被写入本地以下路径：

- `.insightswarm/insightswarm.db`
- `.insightswarm/artifacts/`
- `.insightswarm/.tmp/run-<run_id>/steps.jsonl`

## 冒烟测试

```powershell
python -m insightswarm.cli run smoke "smoke test"
```

运行单元测试：

```powershell
python -m pytest -q
```

验收测试需要真实的模型凭证和脚本化的测试固件（fixtures）。

在配置好凭证后，可显式运行这些测试：

```powershell
python -m pytest -q tests/acceptance
```

## 文档

- [架构设计](docs/architecture.md)
- [本地运行指南](docs/running.md)
- [浏览器安全](docs/browser-safety.md)

## 安全注意事项

BrowserAgent 的设计有意偏向保守。登录、凭证输入、支付、上传/下载、Cookie/存储/请求头访问等高风险操作均需要显式授权或已被直接拦截。浏览器的观测结果属于来源获取材料，不作为正式证据。正式证据始于 Extractor（提取器）从原始文档中创建附带原文引用的正式佐证。

## 项目状态

本项目目前是一个活跃开发中的原型。已知最主要的差距是在服务商故障（如模型配额用尽或速率限制）后的运行恢复能力。尽管基于共享存储的运行时可以保留执行进度，但稳健的恢复与重试策略仍在设计中。
