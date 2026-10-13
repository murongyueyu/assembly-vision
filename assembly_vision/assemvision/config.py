# -*- coding: utf-8 -*-
"""配置加载：YAML 优先，没装 PyYAML 时自动退回 JSON（同名 .json）。

用法：
    cfg = load_config("config/camera.yaml")
    cfg.get("capture.fourcc")            # -> 'MJPG'
    cfg.get("ui.panel_ratio", 0.2)       # 带默认值
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

_UNSET = object()


class Config:
    """给嵌套 dict 加一点便利访问，不改变原始结构。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None, path: str = "") -> None:
        self.data: Dict[str, Any] = data or {}
        self.path = path

    # ---------- 读取 ----------
    def get(self, dotted: str, default: Any = _UNSET) -> Any:
        cur: Any = self.data
        for key in dotted.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                if default is _UNSET:
                    raise KeyError("配置项不存在: %s（文件 %s）" % (dotted, self.path))
                return default
        return cur

    def __getitem__(self, dotted: str) -> Any:
        return self.get(dotted)

    def __contains__(self, dotted: str) -> bool:
        try:
            self.get(dotted)
            return True
        except KeyError:
            return False

    # ---------- 写入（标定脚本用） ----------
    def set(self, dotted: str, value: Any) -> None:
        keys = dotted.split(".")
        cur = self.data
        for key in keys[:-1]:
            nxt = cur.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[key] = nxt
            cur = nxt
        cur[keys[-1]] = value

    def update(self, other: "Config") -> None:
        _deep_update(self.data, other.data)

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("没有指定保存路径")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if path.lower().endswith((".yaml", ".yml")):
            yaml = _import_yaml()
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.data, f, allow_unicode=True,
                               sort_keys=False, default_flow_style=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        return path

    def __repr__(self) -> str:
        return "Config(%s)" % (self.path or "<inline>")


def _import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:
        return None


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def load_config(path: str, required: bool = True) -> Config:
    """加载配置。.yaml 需要 PyYAML；没有就尝试同目录同名 .json。"""
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".json"
        if os.path.exists(alt):
            path = alt
        elif required:
            raise FileNotFoundError("配置文件不存在: %s（也没有 %s）" % (path, alt))
        else:
            return Config({}, path)

    if path.lower().endswith((".yaml", ".yml")):
        yaml = _import_yaml()
        if yaml is None:
            alt = os.path.splitext(path)[0] + ".json"
            if os.path.exists(alt):
                with open(alt, "r", encoding="utf-8") as f:
                    return Config(json.load(f), alt)
            raise ImportError(
                "读取 %s 需要 PyYAML。请执行：pip install PyYAML\n"
                "或者把配置改成同名 .json 文件。" % path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    return Config(data, path)


def merge(*configs: Config) -> Config:
    """后者覆盖前者，用于 camera + workpiece + calibration 三合一。"""
    out = Config({})
    for c in configs:
        if c is not None:
            _deep_update(out.data, c.data)
    return out
