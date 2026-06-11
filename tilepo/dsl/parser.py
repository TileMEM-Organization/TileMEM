from __future__ import annotations

from dataclasses import dataclass
import ast
import re
from typing import Any


REQUIRED_BLOCKS = (
    "model",
    "workload",
    "tile",
    "memory",
    "precision",
    "schedule",
    "autotune",
    "runtime",
)


class DSLParseError(ValueError):
    pass


@dataclass(frozen=True)
class DSLBlock:
    kind: str
    name: str
    values: dict[str, Any]
    line: int


@dataclass(frozen=True)
class DSLPlan:
    blocks: list[DSLBlock]

    def required_block(self, kind: str) -> DSLBlock:
        matches = [block for block in self.blocks if block.kind == kind]
        if not matches:
            raise DSLParseError(f"missing required block: {kind}")
        if len(matches) > 1:
            raise DSLParseError(f"duplicate block kind: {kind}")
        return matches[0]

    def compiled_text(self) -> str:
        lines: list[str] = []
        for block in self.blocks:
            lines.append(f"{block.kind} {block.name} {{")
            for key in sorted(block.values):
                lines.append(f"  {key} = {_format_value(block.values[key])}")
            lines.append("}")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


_HEADER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{\s*$")


def parse_tmem(text: str) -> DSLPlan:
    blocks: list[DSLBlock] = []
    current_kind: str | None = None
    current_name: str | None = None
    current_line = 0
    values: dict[str, Any] = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if current_kind is None:
            match = _HEADER_RE.match(line)
            if not match:
                raise DSLParseError(f"line {lineno}: expected '<block> <name> {{'")
            current_kind, current_name = match.group(1), match.group(2)
            if current_kind not in REQUIRED_BLOCKS:
                raise DSLParseError(f"line {lineno}: unknown block kind '{current_kind}'")
            current_line = lineno
            values = {}
            continue
        if line == "}":
            blocks.append(DSLBlock(current_kind, current_name or "", values, current_line))
            current_kind = None
            current_name = None
            values = {}
            continue
        if "=" not in line:
            raise DSLParseError(f"line {lineno}: expected key = value")
        key, value_text = [part.strip() for part in line.split("=", 1)]
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            raise DSLParseError(f"line {lineno}: invalid key '{key}'")
        values[key] = _parse_value(value_text, lineno)

    if current_kind is not None:
        raise DSLParseError(f"line {current_line}: unterminated block '{current_kind}'")
    plan = DSLPlan(blocks)
    for kind in REQUIRED_BLOCKS:
        plan.required_block(kind)
    return plan


def _strip_comment(line: str) -> str:
    in_string = False
    escaped = False
    result = []
    for char in line:
        if char == "\\" and in_string:
            escaped = not escaped
            result.append(char)
            continue
        if char == '"' and not escaped:
            in_string = not in_string
        escaped = False
        if char == "#" and not in_string:
            break
        result.append(char)
    return "".join(result)


def _parse_value(value_text: str, lineno: int) -> Any:
    if not value_text:
        raise DSLParseError(f"line {lineno}: missing value")
    lowered = value_text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        value = ast.literal_eval(value_text)
    except (SyntaxError, ValueError) as exc:
        raise DSLParseError(f"line {lineno}: invalid value '{value_text}'") from exc
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool, list)):
        return value
    raise DSLParseError(f"line {lineno}: unsupported value type {type(value).__name__}")


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value).replace("'", '"')

