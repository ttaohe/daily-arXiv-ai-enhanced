#!/usr/bin/env python3
import json
import subprocess
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path('/root/.openclaw/workspace/projects/arxiv-daily')
REMOTE = 'myfork'
BRANCH = 'data'
TZ = ZoneInfo('Asia/Shanghai')


def run(cmd):
    return subprocess.run(cmd, cwd=REPO, check=True, text=True, capture_output=True)


def git_show(pathspec):
    try:
        return run(['git', 'show', pathspec]).stdout
    except subprocess.CalledProcessError:
        return ''


def load_jsonl_from_git(pathspec):
    text = git_show(pathspec)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    now = datetime.now(TZ)
    target_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
    prev_date = (now - timedelta(days=2)).strftime('%Y-%m-%d')

    run(['git', 'fetch', '--all', '--prune'])
    try:
        run(['git', 'checkout', 'sync-myfork'])
    except subprocess.CalledProcessError:
        pass
    try:
        run(['git', 'pull', '--ff-only', 'myfork', 'main'])
    except subprocess.CalledProcessError:
        # do not fail whole summary on main branch pull hiccup; data branch fetch is enough
        pass

    current_candidates = [
        f'{target_date}_AI_enhanced_Chinese.jsonl',
        f'{target_date}.jsonl',
    ]
    prev_candidates = [
        f'{prev_date}_AI_enhanced_Chinese.jsonl',
        f'{prev_date}.jsonl',
    ]

    current = []
    current_file = current_candidates[0]
    for name in current_candidates:
        current = load_jsonl_from_git(f'{REMOTE}/{BRANCH}:data/{name}')
        if current:
            current_file = name
            break

    previous = []
    prev_file = prev_candidates[0]
    for name in prev_candidates:
        previous = load_jsonl_from_git(f'{REMOTE}/{BRANCH}:data/{name}')
        if previous:
            prev_file = name
            break

    if not current:
        print(f'今天未在远程 data 分支找到 {target_date} 的数据文件（尝试过 AI 增强版和原始版），可能 action 还没产出，或者今天没有新内容。')
        return

    current_ids = {p.get('id') for p in current if p.get('id')}
    previous_ids = {p.get('id') for p in previous if p.get('id')}
    new_papers = [p for p in current if p.get('id') not in previous_ids]
    compare_pool = new_papers if new_papers else current

    cat_counter = Counter()
    for paper in compare_pool:
        for c in paper.get('categories', [])[:3]:
            cat_counter[c] += 1

    lines = []
    lines.append(f'📚 arXiv 日报（{target_date}，北京时间 9:30 同步）')
    lines.append(f'远程 data 分支已同步，今日数据文件：{current_file}')
    lines.append(f'论文数：{len(current)}；相较前一天新增/变化：{len(new_papers)}')
    if cat_counter:
        top_cats = '、'.join(f'{k}({v})' for k, v in cat_counter.most_common(5))
        lines.append(f'热门分类：{top_cats}')

    focus = compare_pool[:8]
    if focus:
        lines.append('值得关注的论文：')
        for idx, paper in enumerate(focus, 1):
            title = paper.get('title', 'Untitled').strip()
            pid = paper.get('id', '')
            tldr = (((paper.get('AI') or {}).get('tldr')) or '').strip()
            if tldr:
                lines.append(f'{idx}. {title} ({pid}) — {tldr}')
            else:
                lines.append(f'{idx}. {title} ({pid})')

    print('\n'.join(lines))


if __name__ == '__main__':
    main()
