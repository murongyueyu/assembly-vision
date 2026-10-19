# -*- coding: utf-8 -*-
"""尺寸开发候选只能放宽已授权上限，不能关闭必检或修改其他身份条件。"""
import copy
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'tools')]
from prepare_identity_size_candidate import build_candidate
from part_dataset import required_features_match


def main():
    model = {'policy': {'experiment': {'required_features': ['length_mm', 'short_side_mm'],
        'confirmation': {'frames': 7, 'min_score': .85}}},
        'templates': {'P2_column': {'ranges': {'length_mm': [55,77], 'short_side_mm': [6,10.88]}},
                      'P4_cap': {'ranges': {'length_mm': [15.5,23.76], 'short_side_mm': [14.9,22.95]}}}}
    original = copy.deepcopy(model)
    result, changes = build_candidate(model, {'P2_column': {'short_side_mm': 13},
        'P4_cap': {'length_mm': 25, 'short_side_mm': 24}})
    assert model == original and result['policy'] == model['policy'] and len(changes) == 3
    ranges = result['templates']['P2_column']['ranges']
    assert required_features_match({'length_mm':70,'short_side_mm':12.77}, ranges, ['length_mm','short_side_mm'])
    for features in ({'length_mm':70,'short_side_mm':13.01}, {'length_mm':80,'short_side_mm':12},
                     {'length_mm':70,'short_side_mm':5.9}, {'length_mm':70}):
        assert not required_features_match(features, ranges, ['length_mm','short_side_mm'])
    for limits in ({}, {'P2_column': {'length_mm':80}}, {'P2_column': {'short_side_mm':float('inf')}},
                   {'P2_column': {'short_side_mm':True}}, {'P2_column': {'short_side_mm':10}}):
        try:
            build_candidate(model, limits)
        except ValueError:
            pass
        else:
            raise AssertionError('非法或越界候选配置未拒绝')
    print('尺寸候选：三处上限、下限保留、尺寸必检及七帧0.85不变，越界仍拒绝')


if __name__ == '__main__':
    main()
