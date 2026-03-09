import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime

def fetch_arxiv_papers():
    # 你的研究领域关键词
    keywords = [
        'Speculative Decoding', 'KV Cache', 'FSDP', 'Quantization', 
        'DeepSpeed', 'Megatron-LM', 'GPU utilization', 'SLO-aware'
    ]
    
    # 构造查询语句 (搜索摘要和标题)
    query_parts = []
    for kw in keywords:
        query_parts.append(f'ti:"{kw}" OR abs:"{kw}"')
    
    search_query = ' OR '.join(query_parts)
    # 限制在相关分类中：AI, CL, LG, DC (Distributed Computing)
    categories = '(cat:cs.AI OR cat:cs.CL OR cat:cs.LG OR cat:cs.DC OR cat:cs.PF)'
    
    # 完整的查询字符串
    full_query = f"({search_query}) AND {categories}"
    
    params = {
        'search_query': full_query,
        'start': 0,
        'max_results': 50,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    base_url = 'http://export.arxiv.org/api/query?'
    # 正确编码 URL
    url = f"{base_url}{urllib.parse.urlencode(params)}"
    
    print(f"🚀 正在从 arXiv 抓取论文...\nURL: {url}")
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read().decode('utf-8')
        root = ET.fromstring(xml_data)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            paper_id = entry.find('atom:id', ns).text.split('/')[-1]
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
            published = entry.find('atom:published', ns).text.split('T')[0]
            link = entry.find('atom:id', ns).text
            
            # 提取分类
            cats = [cat.get('term') for cat in entry.findall('atom:category', ns)]
            
            papers.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
                "authors": ", ".join(authors),
                "categories": ", ".join(cats),
                "published": published,
                "link": link
            })
            
        print(f"✅ 成功抓取到 {len(papers)} 篇论文！")
        return papers
    except Exception as e:
        print(f"❌ 抓取失败: {e}")
        return []

def main():
    papers = fetch_arxiv_papers()
    if not papers:
        return

    # 按照页面要求生成文件名 (今天日期)
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f"data/{today}_AI_enhanced_Chinese.jsonl"
    
    os.makedirs('data', exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + '\n')
            
    print(f"💾 数据已保存至: {filename}")
    
    # 更新 assets/file-list.txt
    # 列出 data 文件夹下所有的 jsonl 文件
    files = [f for f in os.listdir('data') if f.endswith('.jsonl')]
    files.sort(reverse=True)
    
    with open('assets/file-list.txt', 'w', encoding='utf-8') as f:
        for file in files:
            f.write(f"{file}\n")
    print(f"📝 索引已更新: assets/file-list.txt")

if __name__ == "__main__":
    main()
