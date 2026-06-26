from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from insightswarm.cli import main as cli_main


@dataclass
class RuntimeConfig:
    provider: str = "qwen"
    model: str | None = None
    search_provider: str = "tavily"
    browser_backend: str = "visible"
    browser_cdp_url: str | None = None
    model_config_path: str | None = None
    max_steps: int = 30
    max_runtime_seconds: float = 3600.0
    max_no_progress_seconds: float = 300.0
    max_drain_seconds: float = 1200.0
    input_files: list[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid4().hex[:12])


HELP_TEXT = """\
Commands:
  /ask <question> [--image PATH] [--model NAME] [--json]
  /model <name>                 Set text model for later asks.
  /provider <name>              Set provider, e.g. qwen, dashscope, fake.
  /image <path>                 Attach a default input file for later asks.
  /clear-images                 Clear default input files.
  /config                       Show current runtime config.
  /help                         Show this help.
  /exit                         Exit.

Bare text is treated as /ask <text>.
"""


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent
    argv = list(sys.argv[1:] if argv is None else argv)
    config = RuntimeConfig()
    _initialize_config(repo_root, config)
    _configure_process(repo_root, config)
    if argv:
        line = " ".join(argv)
        if line.startswith("/"):
            return _handle_command(line, config, repo_root)
        return _run_ask(line, config, repo_root, json_output=False, extra_input_files=[], model_override=None)
    return _repl(config, repo_root)


def _repl(config: RuntimeConfig, repo_root: Path) -> int:
    print(_startup_banner(config, repo_root))
    while True:
        try:
            line = input("swarm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if not line:
            continue
        code = _handle_command(line, config, repo_root)
        if code == _EXIT_CODE:
            return 0


_EXIT_CODE = 99


def _handle_command(line: str, config: RuntimeConfig, repo_root: Path) -> int:
    if not line.startswith("/"):
        return _run_ask(line, config, repo_root, json_output=False, extra_input_files=[], model_override=None)
    command, _, rest = line.partition(" ")
    command = command.lower()
    rest = rest.strip()
    if command in {"/exit", "/quit"}:
        return _EXIT_CODE
    if command == "/help":
        print(HELP_TEXT)
        return 0
    if command == "/config":
        _print_config(config)
        return 0
    if command == "/model":
        if not rest:
            print(f"model={config.model or '(unset)'}")
            return 0
        config.model = _unquote(rest)
        print(f"model set to {config.model}")
        return 0
    if command == "/provider":
        if not rest:
            print(f"provider={config.provider}")
            return 0
        config.provider = _unquote(rest)
        print(f"provider set to {config.provider}")
        return 0
    if command == "/image":
        if not rest:
            print("default input files:")
            for path in config.input_files:
                print(f"  {path}")
            return 0
        path = _require_existing_path(_unquote(rest))
        config.input_files.append(path)
        print(f"added input file: {path}")
        return 0
    if command == "/clear-images":
        config.input_files.clear()
        print("default input files cleared")
        return 0
    if command == "/ask":
        return _handle_ask(rest, config, repo_root)
    print(f"unknown command: {command}. Type /help.")
    return 1


def _handle_ask(rest: str, config: RuntimeConfig, repo_root: Path) -> int:
    parser = argparse.ArgumentParser(prog="/ask", add_help=False)
    parser.add_argument("--image", "--input-file", dest="input_files", action="append", default=[])
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--search-provider", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("question", nargs=argparse.REMAINDER)
    try:
        args = parser.parse_args(_split(rest))
    except SystemExit:
        print("usage: /ask <question> [--image PATH] [--model NAME] [--json]")
        return 2
    question = " ".join(args.question).strip()
    if not question:
        print("usage: /ask <question> [--image PATH] [--model NAME] [--json]")
        return 2
    if args.provider:
        config.provider = args.provider
    if args.search_provider:
        config.search_provider = args.search_provider
    extra_files = [_require_existing_path(path) for path in args.input_files]
    return _run_ask(
        question,
        config,
        repo_root,
        json_output=bool(args.json),
        extra_input_files=extra_files,
        model_override=args.model,
    )


def _run_ask(
    question: str,
    config: RuntimeConfig,
    repo_root: Path,
    *,
    json_output: bool,
    extra_input_files: list[str],
    model_override: str | None,
) -> int:
    active_model = model_override or config.model
    if not _has_model_for_provider(config.provider, active_model, config.model_config_path):
        print(
            "No text model is configured. Use `/model <name>` first, "
            "or set INSIGHTSWARM_QWEN_TEXT_MODEL / INSIGHTSWARM_TEXT_MODEL, "
            "or provide a model config path."
        )
        return 2
    _configure_process(repo_root, config)
    cli_args = [
        "--model-provider",
        config.provider,
        "run",
        "ask",
        question,
        "--search-provider",
        config.search_provider,
        "--browser-backend",
        config.browser_backend,
        "--max-steps",
        str(config.max_steps),
        "--max-runtime-seconds",
        str(config.max_runtime_seconds),
        "--max-no-progress-seconds",
        str(config.max_no_progress_seconds),
        "--max-drain-seconds",
        str(config.max_drain_seconds),
    ]
    if config.model_config_path:
        cli_args[0:0] = ["--model-config-path", config.model_config_path]
    if config.browser_cdp_url:
        cli_args.extend(["--browser-cdp-url", config.browser_cdp_url])
    if active_model:
        cli_args.extend(["--model", active_model])
    for path in [*config.input_files, *extra_input_files]:
        cli_args.extend(["--input-file", path])
    if json_output:
        cli_args.append("--json")
    try:
        return int(cli_main(cli_args) or 0)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        print(code)
        return 1


def _configure_process(repo_root: Path, config: RuntimeConfig) -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("INSIGHTSWARM_BROWSER_PROFILE_ROOT", str(repo_root / ".tmp" / "browser-profiles"))


def _has_model_for_provider(provider: str, model: str | None, model_config_path: str | None) -> bool:
    if model_config_path:
        return True
    normalized = str(provider or "").strip().lower()
    if normalized == "fake":
        return True
    provider_key = normalized.upper().replace("-", "_")
    return bool(
        model
        or os.getenv(f"INSIGHTSWARM_{provider_key}_TEXT_MODEL")
        or os.getenv("INSIGHTSWARM_TEXT_MODEL")
        or os.getenv("OPENAI_COMPATIBLE_MODEL")
    )


def _print_config(config: RuntimeConfig) -> None:
    print(f"provider: {config.provider}")
    print(f"model: {_active_model(config)}")
    print(f"search_provider: {config.search_provider}")
    print(f"browser_backend: {config.browser_backend}")
    print(f"browser_profile_root: {os.getenv('INSIGHTSWARM_BROWSER_PROFILE_ROOT')}")
    print(f"model_config_path: {config.model_config_path or '(env/default)'}")
    print(f"max_runtime_seconds: {config.max_runtime_seconds:g}")
    print("default input files:")
    if config.input_files:
        for path in config.input_files:
            print(f"  {path}")
    else:
        print("  none")


def _startup_banner(config: RuntimeConfig, repo_root: Path) -> str:
    width = max(72, min(shutil.get_terminal_size((96, 24)).columns, 118))
    content_width = width - 4
    logo = _wide_logo() if width >= 96 else _compact_logo()
    accent = _ansi("36")
    violet = _ansi("35")
    dim = _ansi("2")
    reset = _ansi("0")
    lines: list[str] = []
    lines.append("+" + "-" * (width - 2) + "+")
    for logo_line in logo:
        rendered = f"{accent}{logo_line}{reset}" if accent else logo_line
        lines.append(_box_line(rendered, content_width, raw_len=len(logo_line)))
    lines.append(_box_line("", content_width))
    subtitle = "local multi-agent research runtime"
    lines.append(_box_line(f"{violet}{subtitle}{reset}" if violet else subtitle, content_width, raw_len=len(subtitle)))
    lines.append(_box_line("", content_width))
    rows = [
        ("workspace", str(repo_root)),
        ("provider", config.provider),
        ("model", _active_model(config)),
        ("search", config.search_provider),
        ("browser", config.browser_backend),
        ("storage", str(repo_root / ".insightswarm" / "insightswarm.db")),
        ("profile", os.getenv("INSIGHTSWARM_BROWSER_PROFILE_ROOT") or str(repo_root / ".tmp" / "browser-profiles")),
        ("branch", _git_branch(repo_root)),
        ("session", config.session_id),
    ]
    for key, value in rows:
        text = f"{key:<10} {value}"
        lines.append(_box_line(text, content_width))
    lines.append(_box_line("", content_width))
    command_text = "/ask <question>   /model <name>   /config   /help   /exit"
    command_rendered = f"{dim}{command_text}{reset}" if dim else command_text
    lines.append(_box_line(command_rendered, content_width, raw_len=len(command_text)))
    lines.append("+" + "-" * (width - 2) + "+")
    return "\n".join(lines)


def _box_line(text: str, content_width: int, *, raw_len: int | None = None) -> str:
    raw_len = len(text) if raw_len is None else raw_len
    clipped, clipped_len = _clip_display(text, raw_len, content_width)
    return "| " + clipped + " " * (content_width - clipped_len) + " |"


def _clip_display(text: str, raw_len: int, width: int) -> tuple[str, int]:
    if raw_len <= width:
        return text, raw_len
    if width <= 3:
        return "", 0
    plain = _strip_ansi(text)
    clipped = plain[: width - 3] + "..."
    return clipped, len(clipped)


def _strip_ansi(text: str) -> str:
    for token in ("\x1b[36m", "\x1b[35m", "\x1b[2m", "\x1b[0m"):
        text = text.replace(token, "")
    return text


def _wide_logo() -> list[str]:
    return [
        "  ___           _       _     _   ____                         ",
        " |_ _|_ __  ___(_) __ _| |__ | |_|  _ \\__      ____ _ _ __ _ __ ___",
        "  | || '_ \\/ __| |/ _` | '_ \\| __| |_) \\ \\ /\\ / / _` | '__| '_ ` _ \\",
        "  | || | | \\__ \\ | (_| | | | | |_|  __/ \\ V  V / (_| | |  | | | | | |",
        " |___|_| |_|___/_|\\__, |_| |_|\\__|_|     \\_/\\_/ \\__,_|_|  |_| |_| |_|",
        "                  |___/                                             ",
    ]


def _compact_logo() -> list[str]:
    return [
        "  ___           _       _     _   ____                         ",
        " |_ _|_ __  ___(_) __ _| |__ | |_|  _ \\__      ____ _ _ __ ___",
        "  | || '_ \\/ __| |/ _` | '_ \\| __| |_) \\ \\ /\\ / / _` | '__| '_ \\",
        " |___|_| |_|___/_|\\__, |_| |_|\\__|_|     \\_/\\_/ \\__,_|_|  |_| |_|",
        "                  |___/                                      ",
    ]


def _active_model(config: RuntimeConfig) -> str:
    config_label = _model_config_label(config.model_config_path)
    if config_label:
        return config_label
    provider_key = str(config.provider or "").upper().replace("-", "_")
    return (
        config.model
        or os.getenv(f"INSIGHTSWARM_{provider_key}_TEXT_MODEL")
        or os.getenv("INSIGHTSWARM_TEXT_MODEL")
        or "(unset)"
    )


def _initialize_config(repo_root: Path, config: RuntimeConfig) -> None:
    model_config = _discover_model_config(repo_root)
    if model_config:
        config.model_config_path = model_config
        if config.provider == "qwen":
            config.provider = "default"


def _discover_model_config(repo_root: Path) -> str | None:
    env_path = os.getenv("INSIGHTSWARM_MODEL_CONFIG")
    if env_path:
        return str(Path(env_path).expanduser())
    local_path = repo_root / "config.models.json"
    if local_path.exists():
        return str(local_path)
    return None


def _model_config_label(model_config_path: str | None) -> str | None:
    if not model_config_path:
        return None
    try:
        data = json.loads(Path(model_config_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"config: {model_config_path}"
    agents = data.get("agents") if isinstance(data, dict) else {}
    providers = data.get("providers") if isinstance(data, dict) else {}
    if not isinstance(agents, dict) or not isinstance(providers, dict):
        return f"config: {Path(model_config_path).name}"
    default_agent = agents.get("default") or agents.get("researcher") or {}
    if not isinstance(default_agent, dict):
        return f"config: {Path(model_config_path).name}"
    provider_name = str(default_agent.get("provider") or "").strip()
    provider = providers.get(provider_name) if provider_name else None
    if not isinstance(provider, dict):
        return f"config: {Path(model_config_path).name}"
    models = provider.get("models")
    model = None
    if isinstance(models, dict):
        model = models.get(default_agent.get("capability") or "text") or models.get("text")
    if model:
        return f"config:{provider_name}/{model}"
    return f"config: {Path(model_config_path).name}"


def _git_branch(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    try:
        text = head.read_text(encoding="utf-8").strip()
    except OSError:
        return "(unknown)"
    if text.startswith("ref:"):
        return text.rsplit("/", 1)[-1]
    return text[:12] if text else "(unknown)"


def _ansi(code: str) -> str:
    if os.getenv("NO_COLOR") or not sys.stdout.isatty():
        return ""
    return f"\x1b[{code}m"


def _split(text: str) -> list[str]:
    return [_unquote(part) for part in shlex.split(text, posix=False)]


def _unquote(text: str) -> str:
    text = str(text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _require_existing_path(path: str) -> str:
    if not Path(path).exists():
        raise SystemExit(f"input file does not exist: {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
