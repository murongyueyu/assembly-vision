# -*- coding: utf-8 -*-
"""ABCDE 启动前自检：逐级核对冻结绑定的文件是否齐、hash 是否对得上。

不改任何文件，只读 + 打印。用法（在树莓派上）：
    ~/miniforge3/envs/assembly/bin/python tools/preflight_pi.py \
        --integration config/abcde_integration_pi_performance_candidate.yaml
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml

OK, BAD, MISS = 'OK  ', 'HASH', 'MISS'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(label: str, path: Path, expect: str | None) -> bool:
    if not path.exists():
        print('%-5s %-46s %s' % (MISS, label, path))
        return False
    if expect is None:
        print('%-5s %-46s %s' % (OK, label, path))
        return True
    got = sha(path)
    good = got == expect
    print('%-5s %-46s %s' % (OK if good else BAD, label, path))
    if not good:
        print('      expect %s' % expect)
        print('      actual %s' % got)
    return good


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--integration', default='config/abcde_integration_pi_performance_candidate.yaml')
    ap.add_argument('--config-dir', default='config')
    args = ap.parse_args()

    cfg_dir = Path(args.config_dir).resolve()
    integ_path = Path(args.integration).resolve()
    if not integ_path.is_absolute():
        integ_path = (Path.cwd() / args.integration).resolve()
    root = Path.cwd().resolve()

    print('== 集成候选 ==')
    if not check('integration', integ_path, None):
        return 1
    integ = yaml.safe_load(integ_path.read_text())

    ok = True
    print('\n== 冻结来源（相对 integration 所在目录）==')
    for key in ('sequence', 'native_reference'):
        target = (integ_path.parent / integ[key]).resolve()
        ok &= check(key, target, integ.get(key + '_sha256'))

    seq = yaml.safe_load(Path((integ_path.parent / integ['sequence']).resolve()).read_text())
    print('\n== E 工序配置 ==')
    seq_path = (integ_path.parent / integ['sequence']).resolve()
    prestart_path = (seq_path.parent / seq['prestart']).resolve()
    ok &= check('prestart', prestart_path, None)

    pre = yaml.safe_load(prestart_path.read_text())
    ident = pre['identity']
    print('\n== D 开工联检 / A 身份冻结 ==')
    for key in ('model', 'policy', 'observer'):
        target = (prestart_path.parent / ident[key]).resolve()
        ok &= check(key, target, ident.get(key + '_sha256'))

    print('\n== 上游冻结 hash（workpiece / camera）==')
    for key in ('workpiece_sha256', 'camera_sha256'):
        name = 'workpiece.yaml' if key.startswith('workpiece') else 'camera.yaml'
        target = cfg_dir / name
        ok &= check(name, target, ident.get(key))

    observer_path = (prestart_path.parent / ident['observer']).resolve()
    obs = yaml.safe_load(observer_path.read_text())
    print('\n== C 观测模板与多方向参考 ==')
    ok &= check('reference_image', (observer_path.parent / obs['reference_image']).resolve(),
                obs.get('reference_sha256'))
    for view in obs.get('reference_views', []):
        ok &= check('view ' + view['image'], (observer_path.parent / view['image']).resolve(),
                    view.get('sha256'))

    print('\n== 其它运行配置 ==')
    for name in ('calibration.yaml', 'part_dataset.yaml'):
        ok &= check(name, cfg_dir / name, None)
    for name in ('tray_observer_runtime.yaml',):
        ok &= check(name, cfg_dir / name, None)

    print('\n== 高度判定参数 ==')
    # ★ 2026-10-28 分阶段测高：有 height_stages 就打印全部阶段，否则回落到旧的单条。
    if integ.get('height_stages') is not None:
        print(yaml.safe_dump(integ.get('height_stages'), allow_unicode=True, sort_keys=False))
    else:
        print(yaml.safe_dump(integ.get('height_assessment'), allow_unicode=True, sort_keys=False))
    print('== 调度参数 ==')
    print(yaml.safe_dump(integ.get('deployment_runtime'), allow_unicode=True, sort_keys=False))
    print('== 时序参数 ==')
    print(yaml.safe_dump(integ.get('deployment_timing'), allow_unicode=True, sort_keys=False))

    print('\n结果：' + ('全部通过' if ok else '存在缺失/不一致，启动前必须修'))
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
