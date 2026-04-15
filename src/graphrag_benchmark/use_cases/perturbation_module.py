import random
from typing import List, Dict, Optional
from graphrag_benchmark.domain.models import (
    ReasoningSubgraph,
    PerturbedSubgraph,
    PerturbationType,
    Triple,
)
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
            triples=self._get_base_triples(subgraph),
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

        # Lấy danh sách tất cả các node ban đầu trước khi xóa
        original_nodes = set()
        for t in base_triples:
            original_nodes.add(t.subject)
            original_nodes.add(t.object)

        # Thực hiện xóa
        for edge_to_remove in edges_to_remove:
            if edge_to_remove in base_triples:
                base_triples.remove(edge_to_remove)

        # Lấy danh sách các node còn lại sau khi xóa
        remaining_nodes = set()
        for t in base_triples:
            remaining_nodes.add(t.subject)
            remaining_nodes.add(t.object)

        # Khôi phục lại TOÀN BỘ các node bị cô lập (cả answer và non-answer)
        # Bắt buộc gán một dummy edge để node cô lập được duy trì trong mảng Triples
        isolated_nodes = original_nodes - remaining_nodes
        for node in isolated_nodes:
            base_triples.append(
                Triple(
                    subject="http://www.wikidata.org/entity/Q_ISOLATED",
                    predicate="http://www.wikidata.org/prop/direct/P_ISOLATED",
                    object=node,
                )
            )

        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=["There is no information in the context to answer."],
            perturbation_type=PerturbationType.BROKEN,
            triples=base_triples,
        )

    def _get_literal_siblings(
        self, literal_str: str, num_siblings: int = 5
    ) -> List[str]:
        import re
        import random
        from datetime import datetime, timedelta

        # Nếu là dạng Date của Wikidata: 1905-03-16T00:00:00Z
        date_regex = r"^[+-]?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        if re.match(date_regex, literal_str):
            try:
                base_date_str = literal_str.strip("+-")[:10]
                base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
                siblings = []
                for _ in range(num_siblings):
                    offset_days = random.randint(-1825, 1825)  # random within 5 years
                    if offset_days == 0:
                        offset_days = 10
                    new_date = base_date + timedelta(days=offset_days)
                    prefix = "-" if literal_str.startswith("-") else ""
                    siblings.append(
                        f"{prefix}{new_date.strftime('%Y-%m-%dT00:00:00Z')}"
                    )
                return siblings
            except Exception:
                pass

        # Nếu là int/float
        try:
            val = float(literal_str)
            siblings = []
            for _ in range(num_siblings):
                offset = (
                    random.uniform(-abs(val) * 0.5, abs(val) * 0.5)
                    if val != 0
                    else random.uniform(-100, 100)
                )
                if offset == 0:
                    offset = val * 0.1 or 1.0
                new_val = val + offset
                if "." not in literal_str:
                    siblings.append(str(int(new_val)))
                else:
                    siblings.append(f"{new_val:.2f}")
            return siblings
        except ValueError:
            # Fallback for plain String literals: Sinh chuỗi giả giữ nguyên định dạng (Shape-preserving Cloaking)
            vowels = "aeiou"
            consonants = "bcdfghjklmnpqrstvwxyz"
            digits = "0123456789"

            siblings = set()
            attempts = 0
            # Cố gắng sinh đủ số lượng mồi nhử khác nhau
            while len(siblings) < num_siblings and attempts < 20:
                attempts += 1
                chars = []
                for c in literal_str:
                    if c.lower() in vowels:
                        new_c = random.choice(vowels)
                        chars.append(new_c.upper() if c.isupper() else new_c)
                    elif c.lower() in consonants:
                        new_c = random.choice(consonants)
                        chars.append(new_c.upper() if c.isupper() else new_c)
                    elif c.isdigit():
                        chars.append(random.choice(digits))
                    else:
                        chars.append(c)

                fake_str = "".join(chars)
                if fake_str != literal_str:
                    siblings.add(fake_str)

            # Trả về các chuỗi đã bị xáo trộn, nếu lỗi nặng không sinh được thì mới gắn (Alt)
            return (
                list(siblings)
                if siblings
                else [literal_str + f" (Alt {i})" for i in range(1, num_siblings + 1)]
            )

    def generate_type_matching(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Type-matching: Bơm các thực thể nối với parent nhưng dẫn đến một vật đồng loại với Answer.
        """
        base_triples = self._get_base_triples(subgraph)

        # Lấy đáp án đầu tiên làm mồi để tìm Type
        answer_node = (
            subgraph.answers[0]
            if subgraph.answers
            else self._get_answer_node(subgraph.gold_path)
        )

        if answer_node:
            # Tìm cạnh cuối cùng thực sự nối với answer_node
            start_node = "http://www.wikidata.org/entity/Q_DUMMY"
            gold_pred = "http://www.wikidata.org/prop/direct/P_DUMMY"

            for t in subgraph.gold_path.triples + base_triples:
                if t.object == answer_node or (
                    subgraph.answers and t.object in subgraph.answers
                ):
                    start_node = t.subject
                    gold_pred = t.predicate
                    break

            # Chọn ngẫu nhiên một relation KHÁC relation dẫn đến đáp án thật
            valid_rels = list(
                set(
                    [
                        t.predicate
                        for t in base_triples
                        if t.subject == start_node and t.predicate != gold_pred
                    ]
                )
            )
            dummy_rel = (
                random.choice(valid_rels)
                if valid_rels
                else "http://www.wikidata.org/prop/direct/P_DUMMY"
            )

            if answer_node.startswith("http://www.wikidata.org/entity/"):
                # Fetch type of answer
                query_type = f"""
                SELECT ?type WHERE {{
                    <{answer_node}> wdt:P31 ?type.
                }} LIMIT 1
                """
                res_type = self.wikidata_client.execute_query(query_type)
                if res_type and "type" in res_type[0]:
                    type_uri = res_type[0]["type"]["value"]
                    # Search for multiple sibling instances
                    query_sibling = f"""
                    SELECT ?sibling WHERE {{
                        ?sibling wdt:P31 <{type_uri}> .
                        FILTER(?sibling != <{answer_node}>)
                    }} LIMIT 5
                    """
                    res_sibs = self.wikidata_client.execute_query(query_sibling)
                    if res_sibs:
                        for row in res_sibs:
                            if "sibling" in row:
                                sibling_uri = row["sibling"]["value"]
                                base_triples.append(
                                    Triple(
                                        subject=start_node,
                                        predicate=dummy_rel,
                                        object=sibling_uri,
                                    )
                                )
            else:
                # Literal matching: generate fake dates/numbers if it's not an Entity
                fake_siblings = self._get_literal_siblings(answer_node, num_siblings=5)
                for fake_sib in fake_siblings:
                    base_triples.append(
                        Triple(subject=start_node, predicate=dummy_rel, object=fake_sib)
                    )

        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.TYPE_MATCHING,
            triples=base_triples,
        )

    def generate_topological(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Topological: Bơm 3 thực thể "Ngôi sao" phổ biến, như USA (Q30) hay English (Q1860).
        Kèm theo các cạnh hop 1 của chúng để đảm bảo chúng có degree cao trong Subgraph.
        """
        base_triples = self._get_base_triples(subgraph)
        answer_node = (
            subgraph.answers[0]
            if subgraph.answers
            else self._get_answer_node(subgraph.gold_path)
        )

        # Gắn decoy hub vào node cha trực tiếp của answer (hop cuối cùng)
        start_node = "http://www.wikidata.org/entity/Q_DUMMY"
        for t in subgraph.gold_path.triples + base_triples:
            if t.object == answer_node or (
                subgraph.answers and t.object in subgraph.answers
            ):
                start_node = t.subject
                break

        # Hardcode a few highly popular hub entities
        hub_nodes = [
            "http://www.wikidata.org/entity/Q30",  # United States
            "http://www.wikidata.org/entity/Q1860",  # English language
            "http://www.wikidata.org/entity/Q142",  # France
            "http://www.wikidata.org/entity/Q145",  # United Kingdom
            "http://www.wikidata.org/entity/Q31",  # Belgium
            "http://www.wikidata.org/entity/Q17",  # Japan
        ]

        chosen_hubs = random.sample(hub_nodes, k=min(3, len(hub_nodes)))

        # Chọn ngẫu nhiên một relation KHÁC relation dẫn đến đáp án thật
        gold_pred = (
            subgraph.gold_path.triples[-1].predicate
            if subgraph.gold_path.triples
            else "http://www.wikidata.org/prop/direct/P_DUMMY"
        )

        valid_rels = list(
            set(
                [
                    t.predicate
                    for t in base_triples
                    if t.subject == start_node and t.predicate != gold_pred
                ]
            )
        )
        dummy_rel = (
            random.choice(valid_rels)
            if valid_rels
            else "http://www.wikidata.org/prop/direct/P_DUMMY"
        )

        for hub in chosen_hubs:
            # 1. Nối câu hỏi tới Hub
            base_triples.append(
                Triple(subject=start_node, predicate=dummy_rel, object=hub)
            )

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
                    if "p" in row and "o" in row:
                        p_val = row["p"]["value"]
                        o_val = row["o"]["value"]
                        base_triples.append(
                            Triple(subject=hub, predicate=p_val, object=o_val)
                        )

        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=subgraph.answers,
            perturbation_type=PerturbationType.TOPOLOGICAL,
            triples=base_triples,
        )

    def generate_swapping(self, subgraph: ReasoningSubgraph) -> PerturbedSubgraph:
        """
        Biến thể Swapping: Thay đổi tất cả các cạnh kết thúc tại Đáp án sang một node sai nhưng cùng nhóm Type.
        Đảm bảo các thay đổi không làm mất node gốc (giống như logic Broken)
        """
        base_triples = self._get_base_triples(subgraph)

        answer_node = (
            subgraph.answers[0]
            if subgraph.answers
            else self._get_answer_node(subgraph.gold_path)
        )
        sibling_uri = None

        if answer_node and subgraph.gold_path.triples:
            if answer_node.startswith("http://www.wikidata.org/entity/"):
                query_type = (
                    f"SELECT ?type WHERE {{ <{answer_node}> wdt:P31 ?type. }} LIMIT 1"
                )
                res_type = self.wikidata_client.execute_query(query_type)
                if res_type and "type" in res_type[0]:
                    type_uri = res_type[0]["type"]["value"]
                    query_sibling = f"SELECT ?sibling WHERE {{ ?sibling wdt:P31 <{type_uri}> . FILTER(?sibling != <{answer_node}>) }} LIMIT 1"
                    res_sib = self.wikidata_client.execute_query(query_sibling)
                    if res_sib and "sibling" in res_sib[0]:
                        sibling_uri = res_sib[0]["sibling"]["value"]
            else:
                # Đối với literal answer
                fake_sibs = self._get_literal_siblings(answer_node, num_siblings=1)
                if fake_sibs:
                    sibling_uri = fake_sibs[0]

        if sibling_uri:
            original_nodes = set()
            for t in base_triples:
                original_nodes.add(t.subject)
                original_nodes.add(t.object)

            edges_to_remove = []
            edges_to_add = []
            if subgraph.answers:
                for t in base_triples:
                    if t.object in subgraph.answers or t.subject in subgraph.answers:
                        edges_to_remove.append(t)
                        new_subj = (
                            sibling_uri if t.subject in subgraph.answers else t.subject
                        )
                        new_obj = (
                            sibling_uri if t.object in subgraph.answers else t.object
                        )
                        edges_to_add.append(
                            Triple(
                                subject=new_subj, predicate=t.predicate, object=new_obj
                            )
                        )

            if not edges_to_remove:
                last_t = subgraph.gold_path.triples[-1]
                edges_to_remove.append(last_t)
                edges_to_add.append(
                    Triple(
                        subject=last_t.subject,
                        predicate=last_t.predicate,
                        object=sibling_uri,
                    )
                )

            for target_edge in edges_to_remove:
                if target_edge in base_triples:
                    base_triples.remove(target_edge)

            base_triples.extend(edges_to_add)

            # Cập nhật ground truth label (answer) mới
            new_answers = [sibling_uri]
        else:
            new_answers = subgraph.answers

        return PerturbedSubgraph(
            question_id=subgraph.question_id,
            answers=new_answers,
            perturbation_type=PerturbationType.SWAPPING,
            triples=base_triples,
        )

    def generate_all_variants(
        self, subgraph: ReasoningSubgraph
    ) -> Dict[PerturbationType, PerturbedSubgraph]:
        return {
            PerturbationType.CLEAN: self.generate_clean(subgraph),
            PerturbationType.BROKEN: self.generate_broken(subgraph),
            PerturbationType.TYPE_MATCHING: self.generate_type_matching(subgraph),
            PerturbationType.TOPOLOGICAL: self.generate_topological(subgraph),
            PerturbationType.SWAPPING: self.generate_swapping(subgraph),
        }
