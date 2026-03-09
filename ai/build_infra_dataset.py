import os
import json
import time
import random
import argparse
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from typing import List, Dict
import requests

CATEGORIES = '(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.DC OR cat:cs.PF OR cat:cs.AR)'

STRONG_INFRA_PHRASES = [
    'kv cache', 'prefix cache', 'cache eviction', 'cache reuse', 'cache compression',
    'prefill', 'decode', 'disaggregation', 'disaggregated serving', 'pd separation',
    'continuous batching', 'request scheduling', 'admission control',
    'tail latency', 'slo', 'throughput', 'latency',
    'paged attention', 'pagedattention', 'flash attention', 'flashattention',
    'vllm', 'tensorrt-llm', 'serverless', 'inference serving', 'serving system', 'inference engine',
    'tensor parallel', 'pipeline parallel', 'model parallel', 'expert parallel',
    'gpu utilization', 'gpu memory', 'memory efficiency', 'moe serving', 'speculative decoding',
    'chunked prefill', 'load balancing', 'expert load imbalance', 'accelerator', 'fpga', 'npu', 'edge inference'
]

WEAK_CONTEXT_PHRASES = [
    'large language model', 'llm', 'language model', 'inference', 'serving', 'deployment',
    'moe', 'expert', 'transformer', 'attention', 'sparse attention', 'quantization', 'pruning'
]

NEGATIVE_PHRASES = [
    'medical image', 'segmentation', 'radiology', 'surgical', 'speech', 'audio', 'asr', 'whisper',
    'essay scoring', 'political', 'drone', 'wifi sensing', 'fetal', 'vision-language', 'vlm',
    'quantum', 'pde', 'orienteering', 'gaze', 'spreadsheet', 'jailbreak', 'patient',
    'news recommendation', 'public opinion', 'respiratory', 'molecule', 'story generation',
    'graph of thoughts', 'agentic ai', 'semantic role labeling'
]

SYSTEM_PROMPT = """
你是一个严格但懂行的 LLM inference optimization 论文筛选器。
目标是保留真正偏以下三类的论文：
A. 服务端推理系统：KV cache、prefill/decode separation、scheduling、SLO、batching、vLLM、distributed inference
B. 模型侧推理优化：speculative decoding、attention加速、MoE剪枝、quantization、prefill/decode优化
C. 端侧/硬件侧推理优化：FPGA/NPU/accelerator/edge inference/memory-bound decode

请特别区分：
- 核心强相关：论文核心贡献直接服务于 LLM inference efficiency / deployment efficiency
- 扩展相关：虽然不是纯系统论文，但明显改善推理成本/延迟/吞吐/内存/部署效率

明确降权或排除：
- VLM / speech / Whisper / ASR / vision-language token pruning
- Graph of Thoughts / agentic reasoning / 路由但非系统级推理优化
- 纯CV/纯NLP任务论文
- 纯评测/纯应用/纯对齐论文
- 与推理效率关系弱的泛方法论文

输出 JSON：{"relevant": true/false, "score": 0-10, "reason": "简短中文理由"}
""".strip()

ARXIV_USER_AGENT = 'daily-arxiv-ai-enhanced/1.0 (GitHub Actions; contact: repo-actions)'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--date', required=True, help='YYYY-MM-DD')
    p.add_argument('--max-results', type=int, default=500)
    p.add_argument('--threshold', type=int, default=7)
    p.add_argument('--output', default='')
    return p.parse_args()


def fetch_url_with_retry(url: str, timeout: int = 120, retries: int = 6) -> bytes:
    last_error = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, headers={'User-Agent': ARXIV_USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                sleep_s = min(90, 8 * attempt + random.uniform(1, 4))
                print(f'arXiv rate limited (attempt {attempt}/{retries}), sleeping {sleep_s:.1f}s...')
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            last_error = e
            sleep_s = min(30, 3 * attempt + random.uniform(0.5, 2))
            print(f'arXiv fetch failed (attempt {attempt}/{retries}): {e}; sleeping {sleep_s:.1f}s...')
            time.sleep(sleep_s)
    raise RuntimeError(f'Failed to fetch arXiv feed after {retries} attempts: {last_error}')


def fetch_candidates(target_date: str, max_results: int) -> List[Dict]:
    params = {
        'search_query': CATEGORIES,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
    }
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode(params)
    xml = fetch_url_with_retry(url, timeout=120, retries=6)
    root = ET.fromstring(xml)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    rows = []
    for e in root.findall('atom:entry', ns):
        published = e.find('atom:published', ns).text.split('T')[0]
        if published != target_date:
            continue
        rows.append({
            'id': e.find('atom:id', ns).text.split('/')[-1],
            'title': ' '.join(e.find('atom:title', ns).text.split()),
            'summary': ' '.join(e.find('atom:summary', ns).text.split()),
            'authors': ', '.join(a.find('atom:name', ns).text for a in e.findall('atom:author', ns)),
            'categories': ', '.join(c.get('term') for c in e.findall('atom:category', ns)),
            'published': published,
            'link': e.find('atom:id', ns).text,
            'comment': ''
        })
    return rows


def keyword_prefilter(rows: List[Dict]) -> List[Dict]:
    kept = []
    for r in rows:
        txt = f"{r['title']} {r['summary']} {r['categories']}".lower()
        strong_hits = [k for k in STRONG_INFRA_PHRASES if k in txt]
        weak_hits = [k for k in WEAK_CONTEXT_PHRASES if k in txt]
        neg_hits = [k for k in NEGATIVE_PHRASES if k in txt]

        keep = False
        if len(strong_hits) >= 1 and len(neg_hits) <= 1:
            keep = True
        if len(strong_hits) >= 2:
            keep = True
        if len(strong_hits) >= 1 and len(weak_hits) >= 1 and len(neg_hits) == 0:
            keep = True
        if ('attention' in txt or 'pruning' in txt or 'quantization' in txt) and ('llm' in txt or 'language model' in txt or 'serving' in txt) and len(neg_hits) == 0:
            keep = True

        if keep:
            r['keyword_hits'] = (strong_hits + weak_hits)[:12]
            kept.append(r)
    return kept


def llm_judge(row: Dict, api_key: str, base_url: str, model: str) -> Dict:
    payload = {
        'model': model,
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"标题: {row['title']}\n摘要: {row['summary']}\n分类: {row['categories']}"}
        ]
    }
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    url = base_url.rstrip('/') + '/chat/completions'
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()['choices'][0]['message']['content'].strip()
    content = content.replace('```json', '').replace('```', '').strip()
    return json.loads(content)


def assign_tier(row: Dict) -> str:
    score = int(row.get('infra_score', 0) or 0)
    txt = f"{row.get('title','')} {row.get('summary','')} {row.get('categories','')}".lower()
    core_signals = [
        'kv cache', 'prefill', 'decode', 'scheduling', 'admission control', 'continuous batching',
        'vllm', 'serving system', 'inference engine', 'serverless', 'tail latency', 'slo',
        'flashprefill', 'streamwise', 'moeless', 'fpga', 'accelerator'
    ]
    hard_extended_exclusions = ['vision-language', 'vlm', 'speech', 'whisper', 'graph of thoughts', 'agentic ai']
    if any(x in txt for x in hard_extended_exclusions):
        return 'discard_candidate'
    if score >= 8 and any(sig in txt for sig in core_signals):
        return 'core_relevant'
    if score >= 9:
        return 'core_relevant'
    return 'extended_relevant'


def save_jsonl(path: str, rows: List[Dict]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    args = parse_args()
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.deepseek.com')
    model = os.environ.get('MODEL_NAME', 'deepseek-chat')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')

    all_rows = fetch_candidates(args.date, args.max_results)
    print(f'candidates on {args.date}: {len(all_rows)}')

    coarse = keyword_prefilter(all_rows)
    print(f'after keyword prefilter: {len(coarse)}')

    final_rows = []
    for i, row in enumerate(coarse, 1):
        try:
            judge = llm_judge(row, api_key, base_url, model)
            relevant = bool(judge.get('relevant', False))
            score = int(judge.get('score', 0) or 0)
            reason = str(judge.get('reason', ''))
            print(f'[{i}/{len(coarse)}] {row["id"]} relevant={relevant} score={score} reason={reason}')
            if relevant and score >= args.threshold:
                row['infra_score'] = score
                row['infra_reason'] = reason
                row['relevance_tier'] = assign_tier(row)
                if row['relevance_tier'] != 'discard_candidate':
                    final_rows.append(row)
        except Exception as e:
            print(f'[{i}/{len(coarse)}] {row["id"]} judge failed: {e}')

    out = args.output or f'../data/{args.date}.jsonl'
    save_jsonl(out, final_rows)
    print(f'final kept: {len(final_rows)} -> {out}')


if __name__ == '__main__':
    main()
