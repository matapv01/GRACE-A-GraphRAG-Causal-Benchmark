import random
from typing import List, Dict, Optional
from graphrag_benchmark.domain.models import ReasoningSubgraph, PerturbedSubgraph, PerturbationType, Triple
from graphrag_benchmark.interfaces.wikidata_api import WikidataClient

class PerturbationModule:
    """
    Adversarial Factory Module.
    Generates causal interventions on the candidate reasoning space.
    """
    def __init__(self, wikidata_client: WikidataClient):
        self.wikidata_client = wikidata_client

    def _get_base_triples(self, subgraph: ReasoningSubgraph) -> List[Triple]:
        # Return a fresh copy of all triples (gold + extra)
        return list(set(subgraph.gold_path.triples + subgraph.extra_triples))

    def _get_answer_node(self, gold_path) -> Optional[str]:
        if not gold_path.triples:
            return None
        # Assume the object of the last triple in gold path is the answer
        return gold_path.triples[-1].object

    def generate_clean(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Original unaltered candidate space. Baseline for comparison.
        """
        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.CLEAN,
            triples=self._get_base_triples(subgraph)
        )

    def generate_broken(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Broken: Xóa TẤT CẢ các cạnh nối trực tiếp với (các) Answer.
        Kiểm tra xem LLM có bị Parametric Memorization hay không.
        """
        base_triples = self._get_base_triples(subgraph)
        
        # Lọc tìm tất cả các cạnh có chứa bất kì đáp án nào (ở cả Subject hoặc Object) để xóa sổ hoàn toàn rễ của nó
        edges_to_remove = []
        if subgraph.answers:
            for t in base_triples:
                if t.object in subgraph.answers or t.subject in subgraph.answers:
                    edges_to_remove.append(t)
        
        # Nếu bộ lọc không bắt được gì (fallback an toàn do URL khác biệt format),
        # ta vẫn cắt bỏ cạnh cuối cùng của nhánh Gold Path để đảm bảo đứt mạch tối thiểu.
        if not edges_to_remove and subgraph.gold_path.triples:
            edges_to_remove.append(subgraph.gold_path.triples[-1])
            
        # Thực hiện xóa
        for edge_to_remove in edges_to_remove:
            if edge_to_remove in base_triples:
                base_triples.remove(edge_to_remove)
            
        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.BROKEN,
            triples=base_triples
        )

    def generate_type_matching(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Type-matching: Bơm các thực thể nối với câu hỏi nhưng dẫn đến một vật đồng loại với Answer.
        """
        base_triples = self._get_base_triples(subgraph)
        
        # Lấy đáp án đầu tiên làm mồi để tìm Type
        answer_node = subgraph.answers[0] if subgraph.answers else self._get_answer_node(subgraph.gold_path)
        
        if answer_node and answer_node.startswith("http://www.wikidata.org/entity/"):
            # Fetch type of answer
            query_type = f"""
            SELECT ?type WHERE {{
                <{answer_node}> wdt:P31 ?type.
            }} LIMIT 1
            """
            res_type = self.wikidata_client.execute_query(query_type)
            if res_type and 'type' in res_type[0]:
                type_uri = res_type[0]['type']['value']
                # Search for multiple sibling instances
                query_sibling = f"""
                SELECT ?sibling WHERE {{
                    ?sibling wdt:P31 <{type_uri}> .
                    FILTER(?sibling != <{answer_node}>)
                }} LIMIT 5
                """
                res_sibs = self.wikidata_client.execute_query(query_sibling)
                if res_sibs:
                    start_node = subgraph.gold_path.triples[0].subject if subgraph.gold_path.triples else "http://www.wikidata.org/entity/Q_DUMMY"
                    # Sao chép y hệt con đường thực tế của Gold Path để nguỵ trang (Camouflage)
                    dummy_rel = subgraph.gold_path.triples[0].predicate if subgraph.gold_path.triples else "http://www.wikidata.org/prop/direct/P31"
                    for row in res_sibs:
                        if 'sibling' in row:
                            sibling_uri = row['sibling']['value']
                            base_triples.append(Triple(subject=start_node, predicate=dummy_rel, object=sibling_uri))

        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.TYPE_MATCHING,
            triples=base_triples
        )

    def generate_topological(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Topological: Bơm 3 thực thể "Ngôi sao" phổ biến, như USA (Q30) hay English (Q1860).
        Kèm theo các cạnh hop 1 của chúng để đảm bảo chúng có degree cao trong Subgraph.
        """
        base_triples = self._get_base_triples(subgraph)
        start_node = subgraph.gold_path.triples[0].subject if subgraph.gold_path.triples else "http://www.wikidata.org/entity/Q_DUMMY"
        
        # Hardcode a few highly popular hub entities
        hub_nodes = [
            "http://www.wikidata.org/entity/Q30",   # United States
            "http://www.wikidata.org/entity/Q1860", # English language
            "http://www.wikidata.org/entity/Q142",  # France
            "http://www.wikidata.org/entity/Q145",  # United Kingdom
            "http://www.wikidata.org/entity/Q31",   # Belgium
            "http://www.wikidata.org/entity/Q17",   # Japan
        ]
        
        chosen_hubs = random.sample(hub_nodes, k=min(3, len(hub_nodes)))
        # Sao chép y hệt cạnh của Gold Path để Model lầm tưởng đây là hướng đi đúng thực tế
        dummy_rel = subgraph.gold_path.triples[0].predicate if subgraph.gold_path.triples else "http://www.wikidata.org/prop/direct/P31"
        
        for hub in chosen_hubs:
            # 1. Nối câu hỏi tới Hub
            base_triples.append(Triple(subject=start_node, predicate=dummy_rel, object=hub))
            
            # 2. Lấy Hop 1 của Hub để bơm degree cao (lấy tối đa 10 cạnh để đại diện)
            query_hop1 = f"""
            SELECT ?p ?o WHERE {{
                <{hub}> ?p ?o .
                FILTER(isIRI(?o))
            }} LIMIT 10
            """
            res_hop1 = self.wikidata_client.execute_query(query_hop1)
            if res_hop1:
                for row in res_hop1:
                    if 'p' in row and 'o' in row:
                        p_val = row['p']['value']
                        o_val = row['o']['value']
                        base_triples.append(Triple(subject=hub, predicate=p_val, object=o_val))
        
        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.TOPOLOGICAL,
            triples=base_triples
        )

    def generate_swapping(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Swapping: Sửa Gold Path: thay đổi tất cả các cạnh kết thúc tại Đáp án sang một node sai nhưng cùng nhóm Type.
        """
        base_triples = self._get_base_triples(subgraph)
        
        # Dùng một answer bất kỳ làm mồi đổi Type
        answer_node = subgraph.answers[0] if subgraph.answers else self._get_answer_node(subgraph.gold_path)
        
        if answer_node and subgraph.gold_path.triples and answer_node.startswith("http://www.wikidata.org/entity/"):
            # Tìm Sibling ngang hàng
            query_type = f"SELECT ?type WHERE {{ <{answer_node}> wdt:P31 ?type. }} LIMIT 1"
            res_type = self.wikidata_client.execute_query(query_type)
            if res_type and 'type' in res_type[0]:
                type_uri = res_type[0]['type']['value']
                query_sibling = f"SELECT ?sibling WHERE {{ ?sibling wdt:P31 <{type_uri}> . FILTER(?sibling != <{answer_node}>) }} LIMIT 1"
                res_sib = self.wikidata_client.execute_query(query_sibling)
                
                if res_sib and 'sibling' in res_sib[0]:
                    sibling_uri = res_sib[0]['sibling']['value']
                    
                    # Xác định mọi đường link có chứa Answer để hoán đổi (cả Subject và Object)
                    edges_to_remove = []
                    edges_to_add = []
                    if subgraph.answers:
                        for t in base_triples:
                            if t.object in subgraph.answers or t.subject in subgraph.answers:
                                edges_to_remove.append(t)
                                
                                # Gắn mồi nối sang Sibling mới
                                new_subj = sibling_uri if t.subject in subgraph.answers else t.subject
                                new_obj = sibling_uri if t.object in subgraph.answers else t.object
                                edges_to_add.append(Triple(subject=new_subj, predicate=t.predicate, object=new_obj))
                    
                    if not edges_to_remove: # Fallback thay thế triple cuối nếu list answer lệch
                        last_t = subgraph.gold_path.triples[-1]
                        edges_to_remove.append(last_t)
                        edges_to_add.append(Triple(subject=last_t.subject, predicate=last_t.predicate, object=sibling_uri))
                        
                    for target_edge in edges_to_remove:
                        if target_edge in base_triples:
                            base_triples.remove(target_edge)
                            
                    base_triples.extend(edges_to_add)

        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.SWAPPING,
            triples=base_triples
        )

    def generate_all_variants(self, subgraph: ReasoningSubgraph) -> Dict[PerturbationType, PerturbedSubgraph]:
        return {
            PerturbationType.CLEAN: self.generate_clean(subgraph),
            PerturbationType.BROKEN: self.generate_broken(subgraph),
            PerturbationType.TYPE_MATCHING: self.generate_type_matching(subgraph),
            PerturbationType.TOPOLOGICAL: self.generate_topological(subgraph),
            PerturbationType.SWAPPING: self.generate_swapping(subgraph),
        }
