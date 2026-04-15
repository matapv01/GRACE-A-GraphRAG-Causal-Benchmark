from typing import List, Tuple, Set
from graphrag_benchmark.domain.models import (
    QuestionData,
    GroundTruthPath,
    Triple,
    ReasoningSubgraph,
)
from graphrag_benchmark.interfaces.wikidata_api import WikidataClient
from graphrag_benchmark.interfaces.embedding_api import EmbeddingClient


class SemanticRetriever:
    """
    Semantic-driven Beam Search Module (The Exam Generator).
    Retrieves the candidate reasoning space (Reasoning Subgraph)
    to establish a fair test environment based on semantics.
    """

    def __init__(
        self, wikidata_client: WikidataClient, embedding_client: EmbeddingClient
    ):
        self.wikidata_client = wikidata_client
        self.embedding_client = embedding_client

    def fetch_1_hop_relations(self, entity_uri: str) -> List[Tuple[str, str, str, str]]:
        """
        Fetches 1-hop outgoing triples from Wikidata for a specific entity along with relation labels.
        Returns: list of (subject, predicate_uri, predicate_label, object_uri)
        """
        # HACK: Using a simplified SPARQL to fetch outbound edges and relationship labels.
        query = f"""
        SELECT ?predicate ?predicateLabel ?object WHERE {{
            <{entity_uri}> ?predicate ?object .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 500
        """
        results = self.wikidata_client.execute_query(query)
        triples_info = []
        for r in results:
            if "predicate" in r and "object" in r:
                p_uri = r["predicate"]["value"]
                o_uri = r["object"]["value"]
                # Try getting the label if available
                p_label = r.get("predicateLabel", {}).get("value", p_uri.split("/")[-1])
                triples_info.append((entity_uri, p_uri, p_label, o_uri))
        return triples_info

    def beam_search(
        self, question: str, start_entities: List[str], hops: int = 2, top_k: int = 5
    ) -> List[Triple]:
        """
        Executes a Semantic-driven Beam Search starting from a list of entities.
        At each hop, expands nodes, scores relation semantics against the question,
        and keeps only the Top-K relations.
        """
        current_entities = set(start_entities)
        collected_triples: List[Triple] = []
        visited_triples: Set[str] = set()

        for hop in range(hops):
            hop_candidates = []

            # Expand current frontier nodes
            for ent in current_entities:
                relations = self.fetch_1_hop_relations(ent)
                hop_candidates.extend(relations)

            if not hop_candidates:
                break

            # Score relations semantically
            relation_labels = [c[2] for c in hop_candidates]
            scores = self.embedding_client.calculate_similarities(
                question, relation_labels
            )

            # Map candidates to their scores
            scored_candidates = list(zip(hop_candidates, scores))
            # Sort descending by score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Retain top-K
            top_relations = scored_candidates[:top_k]

            # Prepare next frontier and save triples
            next_entities = set()
            for cand, score in top_relations:
                s, p_uri, p_label, o_uri = cand
                triple_id = f"{s}_{p_uri}_{o_uri}"
                if triple_id not in visited_triples:
                    visited_triples.add(triple_id)
                    collected_triples.append(
                        Triple(subject=s, predicate=p_uri, object=o_uri)
                    )
                next_entities.add(o_uri)

            current_entities = next_entities

        return collected_triples

    def generate_subgraph(
        self,
        question_data: QuestionData,
        gold_path: GroundTruthPath,
        hop_count: int = 2,
        top_k: int = 10,
    ) -> ReasoningSubgraph:
        """
        Combines the Ground Truth Path with distractors fetched via beam search,
        making the "Candidate Reasoning Space".
        """
        # Assume the starting entity is the subject of the first gold triple
        start_entities = []
        if gold_path.triples:
            start_entities.append(gold_path.triples[0].subject)

        extra_triples = self.beam_search(
            question=question_data.question,
            start_entities=start_entities,
            hops=hop_count,
            top_k=top_k,
        )

        # Filter out triples that are already in the gold path to maintain disjoint extra triples
        gold_signatures = set(
            f"{t.subject}_{t.predicate}_{t.object}" for t in gold_path.triples
        )
        filtered_extra = []
        for t in extra_triples:
            sig = f"{t.subject}_{t.predicate}_{t.object}"
            if sig not in gold_signatures:
                filtered_extra.append(t)

        return ReasoningSubgraph(
            question_id=question_data.id,
            gold_path=gold_path,
            extra_triples=filtered_extra,
            answers=question_data.answers,
        )
