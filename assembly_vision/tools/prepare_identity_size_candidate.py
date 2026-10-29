# -*- coding: utf-8 -*-
"""按显式配置另存尺寸上限候选；保留必检、确认和其他模板字段，不覆盖来源。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assemvision.config import load_config


def build_candidate(model, upper_bounds):
    result = copy.deepcopy(model)
    changes = []
    # ★ 本次授权限定立柱短边和顶盖两边；不提供关闭尺寸必检或修改确认器的入口。
    allowed = {('P2_column', 'short_side_mm'), ('P4_cap', 'length_mm'), ('P4_cap', 'short_side_mm')}
    for part, limits in upper_bounds.items():
        for key, upper in limits.items():
            if (part, key) not in allowed:
                raise ValueError('超出本次尺寸候选范围')
            old = model['templates'][part]['ranges'][key]
            if isinstance(upper, bool) or not isinstance(upper, (int, float)) or not math.isfinite(upper) or upper <= old[1]:
                raise ValueError('尺寸上限必须为有限数并高于旧上限')
            result['templates'][part]['ranges'][key] = [old[0], float(upper)]
            changes.append(dict(part=part, feature=key, before=old, after=[old[0], float(upper)]))
    if not changes:
        raise ValueError('没有尺寸改动')
    return result, changes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'config/identity_size_relaxation_candidate.yaml')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config(str(args.config)).data
    source = (args.config.resolve().parent/cfg['source_model']).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != cfg['source_sha256']:
        raise ValueError('原候选hash不符')
    model = json.loads(source.read_text(encoding='utf-8'))
    candidate, changes = build_candidate(model, cfg['upper_bounds'])
    policy = source.parent/'policy.json'
    if hashlib.sha256(policy.read_bytes()).hexdigest() != candidate['policy_sha256']:
        raise ValueError('原policy hash不符')
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/'model.json'
    path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.output/'policy.json').write_bytes(policy.read_bytes())
    manifest = dict(source=str(source), source_sha256=cfg['source_sha256'],
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), policy_sha256=candidate['policy_sha256'],
        changes=changes, live_accepted=False,
        note='本轮跨模块三次确认后另存；开发图参与选择，不作为独立验收')
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
