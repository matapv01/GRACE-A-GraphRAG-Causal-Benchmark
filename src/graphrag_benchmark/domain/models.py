from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum

class PerturbationType(str, Enum):
    CLEAN = "clean"
    BROKEN = "broken"
    TYPE_MATCHING = "type_matching"
    TOPOLOGICAL = "topological"
    SWAPPING = "swapping"

class ShortcutType(str, Enum):
    FAITHFUL = "Faithful Reasoning"
    MEMORIZATION = "Parametric Memorization"
    TOPOLOGICAL_BIAS = "Topological Bias"
    TYPE_SHORTCUT = "Type-matching Shortcut"
    PRIOR_OVER_CONTEXT = "Prior-over-Context Bias"
    UNKNOWN = "Unknown/Failed"

class Triple(BaseModel):
    subject: str
    predicate: str
    object: str

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))
        
    def __eq__(self, other):
        return (self.subject, self.predicate, self.object) == (other.subject, other.predicate, other.object)

class GroundTruthPath(BaseModel):
    question_id: str
    triples: List[Triple]

class QuestionData(BaseModel):
    id: str
    question: str
    sparql_query: str
    answers: List[str]

class ReasoningSubgraph(BaseModel):
    question_id: str
    answers: List[str] = []
    gold_path: GroundTruthPath
    extra_triples: List[Triple]

class PerturbedSubgraph(BaseModel):
    question_id: str
    answers: List[str] = []
    perturbation_type: PerturbationType
    triples: List[Triple]

class LLMResponseInfo(BaseModel):
    """
    Thông tin chi tiết LLM trả về cho từng Subgraph Context.
    """
    predicted_text: str
    exact_match: bool
    answer_logprobs: List[float] # Logprobs của từng token tạo nên Answers
    p_norm: float # Sinh ra từ exp(mean(logprobs))

class CausalDiagnosisReport(BaseModel):
    """
    Báo cáo đánh giá tổng quan đến chi tiết cho 1 câu hỏi.
    """
    question_id: str
    clean_accuracy: bool
    causal_effect_scores: Dict[PerturbationType, float] # Tính CES so với Clean
    detected_shortcut: ShortcutType
    details: str


