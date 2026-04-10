import argparse
import os
import re
import shutil
from collections import Counter


PATTERN = re.compile(r'(?P<year>\d{4})_(?P<crop>[a-zA-Z]+)_patch_(?P<patch>\d+)_(?P<kind>data|yield)\.npy$')


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Mahya yield data for AgriSTL.')
    parser.add_argument('--src-dir', required=True, help='Source directory such as data/all_samples_30')
    parser.add_argument('--dst-dir', required=True, help='Destination directory such as data/yield_30')
    parser.add_argument('--train-years', nargs='+', type=int, default=[2016, 2017, 2018, 2019, 2020])
    parser.add_argument('--val-years', nargs='+', type=int, default=[2021])
    parser.add_argument('--test-years', nargs='+', type=int, default=[2022, 2023])
    parser.add_argument('--copy', action='store_true', help='Copy files instead of hard-linking them')
    return parser.parse_args()


def ensure_dirs(dst_dir):
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dst_dir, split), exist_ok=True)


def resolve_split(year, args):
    if year in args.train_years:
        return 'train'
    if year in args.val_years:
        return 'val'
    if year in args.test_years:
        return 'test'
    return None


def transfer(src, dst, copy_files):
    if os.path.exists(dst):
        return
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.link(src, dst)


def main():
    args = parse_args()
    ensure_dirs(args.dst_dir)

    stats = Counter()
    names = sorted(os.listdir(args.src_dir))
    for name in names:
        match = PATTERN.fullmatch(name)
        if not match or match.group('kind') != 'data':
            continue

        year = int(match.group('year'))
        split = resolve_split(year, args)
        if split is None:
            continue

        data_src = os.path.join(args.src_dir, name)
        yield_name = name.replace('_data.npy', '_yield.npy')
        yield_src = os.path.join(args.src_dir, yield_name)
        if not os.path.exists(yield_src):
            raise FileNotFoundError(f'Missing paired yield file for {data_src}')

        data_dst = os.path.join(args.dst_dir, split, name)
        yield_dst = os.path.join(args.dst_dir, split, yield_name)
        transfer(data_src, data_dst, args.copy)
        transfer(yield_src, yield_dst, args.copy)

        stats[f'{split}_pairs'] += 1
        stats[f'{split}_{match.group("crop")}'] += 1

    print(f'Prepared dataset from {args.src_dir} to {args.dst_dir}')
    for key in sorted(stats):
        print(f'{key}: {stats[key]}')


if __name__ == '__main__':
    main()
