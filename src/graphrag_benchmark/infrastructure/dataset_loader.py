import json
from typing import List
from graphrag_benchmark.domain.models import QuestionData


class DatasetLoader:
    """Infrastructure detail to load datasets with SPARQL (e.g. LC-QuAD 2.0)."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_dataset(self) -> List[QuestionData]:
        # Implementation depends on the JSON structures for the specific dataset
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both array of questions and {"Questions": [...]} formats
        item_list = data.get("Questions", data) if isinstance(data, dict) else data

        questions = []
        for item in item_list:
            parses = item.get("Parses", [{}])
            sparql = parses[0].get("Sparql", "") if parses else ""

            # Additional support for standard LC-QuAD 2.0 properties
            lcquad_sparql = item.get("sparql_wikidata", item.get("sparql_query", ""))

            q_data = QuestionData(
                id=str(item.get("QuestionId") or item.get("id") or item.get("uid", "")),
                question=str(item.get("RawQuestion") or item.get("question") or ""),
                sparql_query=str(lcquad_sparql or sparql or item.get("sparql", "")),
                answers=item.get("answers", item.get("answer", [])),
            )
            # Only include paths with actual SPARQL
            if q_data.sparql_query and q_data.sparql_query != "None":
                questions.append(q_data)

        return questions
