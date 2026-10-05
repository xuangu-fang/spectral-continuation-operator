"""Read-only research navigation. Never import trainers or load numerical arrays."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "docs/wave_as_prompt/research_catalog.json"
KINDS = ("datasets", "components", "evidence")
ROUTES = {"A", "B", "C"}
FIT = {"adapt", "array_only", "needs_generation", "diagnostic_only", "unsupported", "unverified"}


def repo_path(root: Path, value: str) -> Path:
    """Resolve a repository reference without allowing escape via .. or symlinks."""
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"repository reference must be relative: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"repository reference escapes root: {value}")
    return resolved


def read_catalog(path: Path = DEFAULT_CATALOG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_catalog(catalog: dict, root: Path = ROOT, local_data: bool = False) -> list[str]:
    """Check metadata and path existence only, never read any local data payload."""
    errors = []
    if catalog.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    if set(catalog.get("routes", {})) != ROUTES:
        errors.append("routes must be exactly A/B/C")

    def check_ref(value: str) -> None:
        try:
            path = repo_path(root, value)
            if not path.is_file():
                errors.append(f"missing repository file: {value}")
        except (ValueError, TypeError) as exc:
            errors.append(str(exc))

    for route in catalog.get("routes", {}).values():
        check_ref(route.get("doc", ""))
    for value in catalog.get("core_docs", []):
        check_ref(value)

    forbidden = [
        (Path(p) if Path(p).is_absolute() else root / p).resolve()
        for p in catalog.get("forbidden_input_paths", [])
    ]
    all_ids: set[str] = set()
    for kind in KINDS:
        entries = catalog.get(kind)
        if not isinstance(entries, list):
            errors.append(f"{kind} must be a list")
            continue
        for entry in entries:
            item_id = entry.get("id", "")
            if not item_id or item_id in all_ids:
                errors.append(f"missing or duplicate id: {item_id}")
            all_ids.add(item_id)
            for field in ("name", "status"):
                if not entry.get(field):
                    errors.append(f"{item_id}: missing {field}")
            routes = entry.get("routes", [])
            if not routes or not set(routes) <= ROUTES or len(set(routes)) != len(routes):
                errors.append(f"{item_id}: invalid routes")
            if not entry.get("refs"):
                errors.append(f"{item_id}: missing evidence/code references")
            for ref in entry.get("refs", []):
                check_ref(ref)
            if kind != "datasets":
                continue
            for field in ("local_paths", "shape", "split", "limits"):
                if not entry.get(field):
                    errors.append(f"{item_id}: missing {field}")
            fit = entry.get("fit", {})
            if set(fit) != ROUTES or any(v not in FIT for v in fit.values()):
                errors.append(f"{item_id}: invalid route fit")
            for value in entry.get("local_paths", []):
                path = Path(value)
                if not path.is_absolute():
                    try:
                        path = repo_path(root, value)
                    except ValueError as exc:
                        errors.append(f"{item_id}: {exc}")
                        continue
                path = path.resolve()
                if any(path == deny or path.is_relative_to(deny) for deny in forbidden):
                    errors.append(f"{item_id}: forbidden input path: {value}")
                if local_data and not path.exists():
                    errors.append(f"{item_id}: missing local data: {value}")
    return errors


def markdown_link_errors(paths: list[str], root: Path = ROOT) -> list[str]:
    """Check explicit local Markdown targets in current docs, excluding code blocks."""
    errors = []
    fence = chr(96) * 3
    for value in paths:
        path = repo_path(root, value)
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        content = re.sub(
            r"(?ms)^" + fence + r".*?^" + fence + r"\s*$",
            "",
            path.read_text(encoding="utf-8"),
        )
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", content):
            target = target.strip()
            if target.startswith("<"):
                target = target[1:].split(">", 1)[0]
            else:
                target = target.split(' "', 1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local = (path.parent / unquote(parsed.path)).resolve()
            if not local.exists():
                errors.append(f"{value}: missing link target {target}")
    return errors


def selected(catalog: dict, kind: str | None, route: str | None) -> list[tuple[str, dict]]:
    return [
        (group, entry)
        for group in (KINDS if kind is None else (kind,))
        for entry in catalog[group]
        if route is None or route in entry["routes"]
    ]


def inventory(catalog: dict, root: Path = ROOT) -> str:
    """Produce an index, not a relocation; only inspect text/code filenames."""
    reusable = {ref for entry in catalog["components"] for ref in entry["refs"]}
    current = set(catalog["core_docs"])
    sections = (
        ("文档：当前入口、研究记忆与历史技术细节", "docs", {".md"}),
        ("结果报告：历史证据", "reports", {".md"}),
        ("代码：复用候选或历史复现", "src/spectral_continuation", {".py", ".md"}),
        ("配置：历史冻结协议", "configs", {".json", ".md"}),
        ("维护工具：不作为实验运行计划", "scripts", {".py", ".md"}),
    )
    lines = [
        "# 完整文件清单（2026-10-05 重整）",
        "",
        "> 自动生成的导航，不移动原文件。代码“复用候选”不代表新路线已完成。",
        "> 当前入口见 [三条路线](../wave_as_prompt/README.md)。",
        "> 未列为当前入口的旧计划不授权继续运行。大型数组/checkpoint 不在此索引中。",
        "",
    ]
    for title, directory, suffixes in sections:
        paths = sorted(
            p
            for p in (root / directory).rglob("*")
            if p.is_file()
            and p.suffix in suffixes
            and "__pycache__" not in p.parts
            and p.name != "INVENTORY_20261005.md"
        )
        lines += [f"## {title}（{len(paths)} 项）", ""]
        for path in paths:
            rel = path.relative_to(root).as_posix()
            if rel in current:
                role = "当前入口"
            elif rel in reusable:
                role = "复用候选"
            elif "20261005_before_three_routes/" in rel:
                role = "重整前快照"
            else:
                role = "历史/参考"
            lines.append(f"- [{rel}](../../{rel}) — {role}")
        lines.append("")
    lines += [
        "## 实验产物",
        "",
        "产物不按文件迁移。按原报告中的 artifact 路径、配置和摘要追溯：",
        "[结果入口](../../reports/README.md) / [产物说明](../../artifacts/README.md) /",
        "[实验账本](../EXPERIMENT_LEDGER_ZH.md)。",
        "数据的本地路径、限制和引用由 research_catalog.json 维护；check --local-data 只检查存在性。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list route-specific assets and warnings")
    listing.add_argument("--kind", choices=KINDS)
    listing.add_argument("--route", choices=sorted(ROUTES))
    listing.add_argument("--json", action="store_true")
    checking = sub.add_parser("check", help="check catalog and current documentation links")
    checking.add_argument(
        "--local-data", action="store_true", help="stat local paths; no array reads"
    )
    sub.add_parser("inventory", help="print Markdown inventory to stdout; never write files")
    args = parser.parse_args(argv)
    try:
        catalog = read_catalog(args.catalog)
        errors = validate_catalog(catalog, local_data=getattr(args, "local_data", False))
        if args.command == "check":
            errors += markdown_link_errors(catalog.get("core_docs", []))
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        if args.command == "check":
            counts = ", ".join(f"{key}={len(catalog[key])}" for key in KINDS)
            print(f"OK: {counts}; catalog paths and current Markdown targets checked.")
            print("Local paths stat-checked." if args.local_data else "Local data not checked.")
            print(
                "No arrays loaded; no training or GPU work. File existence is not data validation."
            )
        elif args.command == "inventory":
            print(inventory(catalog), end="")
        else:
            entries = selected(catalog, args.kind, args.route)
            if args.json:
                print(
                    json.dumps(
                        [dict(kind=k, **v) for k, v in entries], ensure_ascii=False, indent=2
                    )
                )
            else:
                for kind, entry in entries:
                    print(f"[{kind}] {entry['id']} {entry['name']} ({entry['status']})")
                    if args.route and "fit" in entry:
                        print(f"  route {args.route}: {entry['fit'][args.route]}")
                    for warning in entry.get("limits", []):
                        print(f"  ! {warning}")
                    if entry.get("note"):
                        print(f"  {entry['note']}")
                    for ref in entry["refs"]:
                        print(f"  -> {ref}")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
