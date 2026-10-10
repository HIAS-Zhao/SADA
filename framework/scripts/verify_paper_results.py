"""Recalculate the released three-seed framework table from group-level CSVs."""

import argparse
import csv
import statistics
from pathlib import Path

FACTORS = (.1, .25, .5, .75, 1., 2., 4.)
FAMILIES = {'qwen25': 5, 'qwen35': 2, 'remoteclip': 1, 'resnet18': 1}
METHODS = ('NoRetrain', 'Periodic', 'Static', 'EWC', 'SADA')
SEEDS = (42, 43, 44)


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def verify(directory):
    directory = Path(directory)
    groups = read(directory / 'qwen25_分组结果.csv') + read(directory / 'omniearth_分组结果.csv')
    per_seed = read(directory / '逐种子结果.csv')
    aggregate = read(directory / '三种子统计.csv')
    if (len(groups), len(per_seed), len(aggregate)) != (945, 420, 140):
        raise ValueError('Unexpected group, seed, or aggregate row count')
    for family, group_count in FAMILIES.items():
        for method in METHODS:
            for factor in FACTORS:
                scores = []
                for seed in SEEDS:
                    subset = [r for r in groups if r['模型'] == family and r['方法'] == method
                              and int(r['种子']) == seed and float(r['负载倍数']) == factor]
                    if len(subset) != group_count:
                        raise ValueError(f'Missing group: {family}, {method}, {factor}, {seed}')
                    denominator = sum(int(r['测试样本数']) for r in subset)
                    score = sum(float(r['窗口准确率']) * int(r['测试样本数']) for r in subset) / denominator
                    scores.append(score)
                    recorded = [r for r in per_seed if r['模型'] == family and r['方法'] == method
                                and float(r['负载倍数']) == factor and int(r['种子']) == seed]
                    if len(recorded) != 1 or abs(float(recorded[0]['窗口准确率']) - score) > 1e-12:
                        raise ValueError(f'Seed score mismatch: {family}, {method}, {factor}, {seed}')
                recorded = [r for r in aggregate if r['模型'] == family and r['方法'] == method
                            and float(r['负载倍数']) == factor]
                if len(recorded) != 1 or abs(float(recorded[0]['均值']) - statistics.mean(scores)) > 1e-12 \
                        or abs(float(recorded[0]['样本标准差']) - statistics.stdev(scores)) > 1e-12:
                    raise ValueError(f'Aggregate mismatch: {family}, {method}, {factor}')
    return {'group_rows': len(groups), 'per_seed_cells': len(per_seed), 'aggregate_cells': len(aggregate)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'paper_results')
    print(verify(parser.parse_args().results))
