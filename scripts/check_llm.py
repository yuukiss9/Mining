"""外部服务连通性自检 —— 分三层暴露失败点，让"静默降级"无处可藏。

为什么必须单独写这个脚本：降级设计的副作用是**配置写错时系统不会报错**，
而是安静地走规则分支。你会以为在用 GPT-4o / Qwen，其实一次调用都没发生。
所以要把三层都打出来：

1. **配置层** —— 读到的 base_url / 脱敏 key / 模型名，证明配置真的生效了
   （而不是被环境变量或别处覆盖）。
2. **可达层** —— 对真实端点发一次最小请求，把失败翻译成可照做的方向
   （DNS / 连接被拒 / 401 鉴权 / 404 路径 / 429 限流 / 模型名不匹配）。
3. **调用层** —— 真发一次业务请求，证明鉴权与参数都对。

用法::

    python scripts/check_llm.py            # 三层全查
    python scripts/check_llm.py --quick    # 只查配置层 + 可达层，不消耗额度
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ni101.config import ConfigError, load_config  # noqa: E402
from ni101.llm import build_llm  # noqa: E402

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

#: HTTP 状态码 -> 可照做的排查方向
_STATUS_HINTS: dict[int, str] = {
    200: "端点可达，鉴权与参数均正常",
    400: "端点可达；请求体不合法（本地拼装问题，不是网络问题）",
    401: "端点可达，但鉴权失败 -> 检查 api_key",
    403: "端点可达，但被拒绝 -> key 权限不足或 IP 白名单",
    404: "端点可达，但路径不存在 -> base_url 大概率少了 /v1 或路径写错",
    405: "端点可达，但方法不允许 -> base_url 指向的是网页而不是 API",
    422: "端点可达；请求参数不合法",
    429: "端点可达，但被限流 -> 降低并发或稍后重试",
}


def check_config_layer(label: str, spec) -> bool:
    print(f"\n--- [{label}] 第 1 层：配置 ---")
    print(f"  label      : {spec.label}")
    print(f"  model      : {spec.model or '<empty>'}")
    print(f"  base_url   : {spec.base_url or '<empty>'}")
    print(f"  api_key    : {spec.masked_key()}")
    print(f"  timeout_s  : {spec.timeout_s}")

    if spec.configured:
        print(f"  {PASS} 配置完整（api_key / base_url / model 三项齐全）")
        return True

    missing = ", ".join(spec.missing)
    print(f"  {WARN} 未配置（缺少：{missing}）-> 该环节将走规则降级路径")
    return False


def check_reachability(label: str, spec) -> bool:
    print(f"--- [{label}] 第 2 层：可达性 ---")
    if not spec.configured:
        print(f"  {WARN} 跳过（未配置）")
        return False

    parsed = urlparse(spec.base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        print(f"  {FAIL} base_url 解析不出 host：{spec.base_url}")
        return False

    # 先做极短超时的 TCP 预检：比让 SDK 干等十几秒便宜得多
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"  {PASS} TCP 可达 {host}:{port}")
    except socket.gaierror as exc:
        print(f"  {FAIL} DNS 解析失败（{host}）：{exc} -> 检查 base_url 拼写")
        return False
    except ConnectionRefusedError as exc:
        print(f"  {FAIL} 连接被拒 {host}:{port}：{exc} -> 端口不通或服务未启动")
        return False
    except (TimeoutError, socket.timeout):
        print(f"  {FAIL} 连接超时 {host}:{port} -> 网络不通或需要代理")
        return False
    except OSError as exc:
        print(f"  {FAIL} 网络层失败：{exc}")
        return False

    # HTTP 层：拿真实会被调用的端点探，不用 /models（不少兼容层根本没这条路径）
    url = spec.base_url.rstrip("/") + "/chat/completions"
    try:
        import httpx

        response = httpx.post(
            url,
            headers={"Content-Type": "application/json", "Authorization": "Bearer probe"},
            json={"model": spec.model, "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=10.0,
        )
        hint = _STATUS_HINTS.get(response.status_code, "")
        # HTTP 层有应答即算可达（400/401/403 都说明主机在线）
        print(f"  {PASS} HTTP {response.status_code} @ {url}")
        if hint:
            print(f"         -> {hint}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  {FAIL} HTTP 层无应答：{type(exc).__name__}: {exc}")
        return False


def check_call_layer(label: str, spec, retries: int) -> bool:
    print(f"--- [{label}] 第 3 层：真实调用 ---")
    if not spec.configured:
        print(f"  {WARN} 跳过（未配置）")
        return False

    client = build_llm(spec, retries=retries, verbose=True)
    result = client.probe(quick=False)
    if result.ok:
        preview = result.text.strip().replace("\n", " ")[:60]
        print(f"  {PASS} 调用成功（{result.latency_ms} ms）返回：{preview!r}")
        if result.prompt_tokens is not None:
            print(f"         tokens: prompt={result.prompt_tokens} completion={result.completion_tokens}")
        return True

    print(f"  {FAIL} 调用失败：{result.error}")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM 端点连通性三层自检")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--quick", action="store_true", help="只探配置与可达性，不消耗额度")
    args = parser.parse_args(argv)

    print("=" * 74)
    print("LLM 端点连通性自检（配置层 -> 可达层 -> 调用层）")
    print("=" * 74)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"{FAIL} 配置读取失败：{exc}")
        return 2

    print(f"配置文件：{config.config_path}")
    print("（配置优先级：环境变量 NI101_<AGENT>_<FIELD> > config.yaml）")

    outcomes: list[tuple[str, bool, bool, bool]] = []
    for spec in (config.extractor, config.critic):
        label = spec.label or "llm"
        configured = check_config_layer(label, spec)
        reachable = check_reachability(label, spec)
        called = check_call_layer(label, spec, config.runtime.request_retries) if not args.quick else False
        outcomes.append((label, configured, reachable, called))

    print("\n" + "=" * 74)
    print("结论")
    print("=" * 74)
    failures = 0
    for label, configured, reachable, called in outcomes:
        if not configured:
            print(f"- {label}: 未配置 -> 走规则降级路径（不是错误，但效果弱于真模型）")
            continue
        if not reachable:
            failures += 1
            print(f"- {label}: 配置完整但**不可达** -> 会被静默降级，请先修网络/地址")
            continue
        if args.quick:
            print(f"- {label}: 配置完整且可达（--quick 模式未做真实调用）")
            continue
        if not called:
            failures += 1
            print(f"- {label}: 可达但**调用失败** -> 见上方第 3 层原因")
            continue
        print(f"- {label}: {PASS} 配置、连通、调用三层全部通过")

    print()
    if failures:
        print(f"存在 {failures} 处问题，请按上方提示修复。")
        return 1
    print("未发现问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
