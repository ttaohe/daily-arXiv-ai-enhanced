import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime

# ================= 配置区 =================
DEEPSEEK_API_KEY = "sk-57a42dced13842a787d39d0b6184dd2f" # 已填入你之前提供的 Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# 扩充后的专业词库
LLM_INFRA_KEYWORDS = [
    'Speculative Decoding', 'KV Cache', 'FSDP', 'Quantization', 
    'DeepSpeed', 'Megatron-LM', 'GPU utilization', 'SLO-aware',
    'Prefill-Decoding Disaggregation', 'PD separation', 'Continuous Batching',
    'PagedAttention', 'vLLM', 'Model Parallelism', 'Tensor Parallelism',
    'Pipeline Parallelism', 'Inference Optimization', 'Serving System',
    'Memory Efficiency', 'Throughput Optimization', 'Latency Reduction'
]

ARXIV_CATEGORIES = '(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.DC OR cat:cs.PF OR cat:cs.AR)'
# ==========================================

def call_ai_filter(title, abstract):
    """调用 DeepSeek 评估论文相关性"""
    prompt = f"""
    你是一个大模型推理与系统（LLM Infra）领域的专家。请根据以下论文的标题和摘要，判断其是否属于 LLM 推理优化、分布式训练、显存管理、系统调度、GPU 加速等 Infra 领域。
    如果是，请给出 0-10 的评分（10 分最高），并简短说明理由（10字以内）。
    
    标题: {title}
    摘要: {abstract}
    
    请严格按照 JSON 格式返回：{{"score": 评分, "reason": "原因"}}
    """
    
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    
    req = urllib.request.Request(DEEPSEEK_BASE_URL)
    req.add_header('Content-Type', 'application/json')
    req.add_header('Authorization', f'Bearer {DEEPSEEK_API_KEY}')
    
    try:
        response = urllib.request.urlopen(req, json.dumps(data).encode('utf-8'), timeout=10)
        result = json.loads(response.read().decode('utf-8'))
        content = result['choices'][0]['message']['content']
        # 兼容处理可能带有的 markdown 标签
        content = content.replace('```json', '').replace('```', '').strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ AI 评分失败 ({title[:30]}...): {e}")
        return {"score": 5, "reason": "Error"}

def fetch_all_candidate_papers():
    """抓取候选论文池"""
    query_parts = [f'ti:"{kw}" OR abs:"{kw}"' for kw in LLM_INFRA_KEYWORDS]
    search_query = f"({' OR '.join(query_parts)}) AND {ARXIV_CATEGORIES}"
    
    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': 100, # 扩大搜索池
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    print(f"🚀 正在扫描 arXiv 论文池...")
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read().decode('utf-8')
        root = ET.fromstring(xml_data)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        for entry in root.findall('atom:entry', ns):
            papers.append({
                "id": entry.find('atom:id', ns).text.split('/')[-1],
                "title": entry.find('atom:title', ns).text.strip().replace('\n', ' '),
                "summary": entry.find('atom:summary', ns).text.strip().replace('\n', ' '),
                "authors": ", ".join([a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]),
                "categories": ", ".join([c.get('term') for c in entry.findall('atom:category', ns)]),
                "published": entry.find('atom:published', ns).text.split('T')[0],
                "link": entry.find('atom:id', ns).text
            })
        return papers
    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        return []

def main():
    raw_papers = fetch_all_candidate_papers()
    if not raw_papers: return

    print(f"🔍 原始抓取到 {len(raw_papers)} 篇。开始由 DeepSeek 进行 AI 语义筛选...")
    
    filtered_papers = []
    for paper in raw_papers:
        res = call_ai_filter(paper['title'], paper['summary'])
        if res['score'] >= 7:
            paper['title'] = f"[{res['score']}分] {paper['title']}"
            paper['ai_reason'] = res['reason']
            filtered_papers.append(paper)
            print(f"✅ 保留: {paper['title']} (原因: {res['reason']})")
        else:
            print(f"⏩ 过滤: {paper['title'][:50]}... (得分: {res['score']})")

    # 保存
    today = datetime.now().strftime('%Y-%m-%d')
    # 我们生成两份文件，一份是全量，一份是 AI 筛选后的
    all_filename = f"data/{today}_all.jsonl"
    filtered_filename = f"data/{today}_filtered.jsonl"
    
    os.makedirs('data', exist_ok=True)
    
    with open(all_filename, 'w', encoding='utf-8') as f:
        for p in raw_papers: f.write(json.dumps(p, ensure_ascii=False) + '\n')
    
    with open(filtered_filename, 'w', encoding='utf-8') as f:
        for p in filtered_papers: f.write(json.dumps(p, ensure_ascii=False) + '\n')
            
    print(f"\n📊 筛选完成！原始: {len(raw_papers)} 篇，AI 选中: {len(filtered_papers)} 篇。")
    
    # 更新索引，优先展示 filtered
    files = [f for f in os.listdir('data') if f.endswith('.jsonl')]
    files.sort(reverse=True)
    with open('assets/file-list.txt', 'w', encoding='utf-8') as f:
        for file in files: f.write(f"{file}\n")

if __name__ == "__main__":
    main()
