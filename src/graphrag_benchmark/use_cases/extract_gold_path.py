from typing import List, Optional
import re
from graphrag_benchmark.domain.models import Triple, GroundTruthPath, QuestionData
from graphrag_benchmark.interfaces.wikidata_api import WikidataClient

class WikidataExtractor:
    """
    Module responsible for extracting the Ground Truth ('Sự thật') / Gold Path 
    for an input question based on the provided SPARQL query.
    """
    def __init__(self, wikidata_client: WikidataClient):
        self.wikidata_client = wikidata_client

    def extract_gold_path(self, question: QuestionData) -> GroundTruthPath:
        original_query = question.sparql_query
        
        # 1. Trích xuất mệnh đề WHERE để lấy cấu trúc thô của các Triple (Bẫy Gold Path)
        where_match = re.search(r'WHERE\s*\{([\s\S]*?)\}', original_query, re.IGNORECASE)
        if not where_match:
            return GroundTruthPath(question_id=question.id, triples=[])
            
        where_clause = where_match.group(1).strip()
        
        # Phân tích thô các triple patterns cách nhau bởi dấu chấm '.'
        # Ví dụ: wd:Q169794 wdt:P26 ?X . ?X wdt:P22 ?answer
        pattern_parts = [p.strip() for p in where_clause.split('.') if p.strip()]
        
        triple_patterns = []
        for part in pattern_parts:
            tokens = part.split()
            if len(tokens) == 3: # Subject Predicate Object
                triple_patterns.append(tokens)

        # 2. Đổi SELECT * để lấy được toàn bộ mapping của các biến (ví dụ: ?X, ?answer, ?obj)
        select_star_query = original_query
        
        # Bắt tên biến đang được SELECT (như ?obj, ?answer) để lấy làm đáp án đích
        target_var = None
        var_match = re.search(r'SELECT\s+(?:DISTINCT\s+)?(\?\w+)', original_query, re.IGNORECASE)
        if var_match:
            target_var = var_match.group(1)[1:] # ví dụ '?obj' -> 'obj'
            
        # Thay thế mệnh đề SELECT (có hoặc không có DISTINCT) và các biến -> SELECT *
        select_star_query = re.sub(r'SELECT\s+(?:DISTINCT\s+)?(.*?)\s+WHERE', 'SELECT * WHERE', original_query, flags=re.IGNORECASE)
            
        try:
            results = self.wikidata_client.execute_query(select_star_query)
        except Exception:
            # Nếu truy vấn lỗi, trả về rỗng
            return GroundTruthPath(question_id=question.id, triples=[])
        
        # Bổ sung đáp án tìm thấy vào mảng chứa của Dataset
        if target_var and not question.answers:
            answers_set = set()
            for row in results:
                if target_var in row:
                    val = row[target_var]['value']
                    answers_set.add(val)
            question.answers = list(answers_set)
        
        # 3. Thay thế các biến trong Triple pattern bằng giá trị thực tế sinh ra từ Wikidata
        triples = []
        for row in results:
            for pattern in triple_patterns:
                s, p, o = pattern
                
                # Hàm helper để map Prefix thành URI hoặc lấy giá trị biến
                def resolve_token(token):
                    if token.startswith('?'):
                        var_name = token[1:]
                        if var_name in row: return row[var_name]['value']
                    elif token.startswith('wd:'): return token.replace('wd:', 'http://www.wikidata.org/entity/')
                    elif token.startswith('wdt:'): return token.replace('wdt:', 'http://www.wikidata.org/prop/direct/')
                    elif token.startswith('p:'): return token.replace('p:', 'http://www.wikidata.org/prop/')
                    elif token.startswith('<') and token.endswith('>'): return token[1:-1]
                    return token
                
                s_resolved = resolve_token(s)
                p_resolved = resolve_token(p)
                o_resolved = resolve_token(o)
                
                # Chỉ thêm nếu tất cả đã được phân giải thành URI hợp lệ
                if '?' not in s_resolved and '?' not in p_resolved and '?' not in o_resolved:
                    triple = Triple(subject=s_resolved, predicate=p_resolved, object=o_resolved)
                    # Deduplicate in current list
                    if triple not in triples:
                        triples.append(triple)
                
        return GroundTruthPath(
            question_id=question.id,
            triples=triples
        )
