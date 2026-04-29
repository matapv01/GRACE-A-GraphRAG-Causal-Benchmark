import re
import asyncio
import math
from typing import Dict, List, Tuple, Tuple
import numpy as np
from graphrag_benchmark.domain.models import (
    PerturbationType,
    ShortcutType,
    CausalDiagnosisReport,
    LLMResponseInfo,
)


import os
from openai import AsyncOpenAI
# =========================================================================

class EvaluationModule:
    """
    Evaluate LLM faithfully against causal shortcuts.
    Từ tổng quan (Accuracy) đến bóc tách chi tiết (Token Logprobs -> P_norm -> CES).
    """

    def compute_exact_match(self, predicted_text: str, ground_truths: List[str]) -> float:
        """
        Dành cho các type: CLEAN, TYPE_MATCHING, TOPOLOGICAL, SWAPPING.
        Đánh giá bằng metric đã build (String Inclusion / Exact Match).
        """
        pred_lower = predicted_text.lower()
        for gt in ground_truths:
            if gt.lower() in pred_lower:
                return 1.0
        return 0.0

    async def compute_llm_broken_score(self, predicted_text: str, broken_ground_truth: str) -> float:
        """
        Dành riêng cho type: BROKEN trong bài toán MCQ ABCD.
        Hệ thống RAG phải trả về 'None' để chứng minh là nó không bịa đặt khi thiếu context.
        """
        text = predicted_text.strip()
        
        # Nếu model ngoan ngoãn trả ra chữ None thật sự hoặc có None trong chuỗi
        if text.lower() == "none" or "none" in text.lower():
            return 1.0
        
        return 0.0

    async def compute_llm_based_score(self, predicted_text: str, ground_truths: List[str], variant: str, global_labels: dict, question: str = "") -> Tuple[float, str]:
        """
        Regex/Heuristic làm Judge để chấm điểm MCQ thay vì dùng LLM tốn kém.
        Dành cho bài toán Trắc nghiệm ABCD: Đối chiếu chữ cái mà RAG sinh ra với chữ cái đáp án đúng.
        Trả về (Score_Float, Chữ_Cái_Dự_Đoán)
        """
        expected_letter = (ground_truths[0] if ground_truths else "None").upper()
        text = str(predicted_text or "")
        text_upper = text.upper()

        # Ưu tiên các mẫu "kết luận đáp án" rõ ràng để tránh match nhầm chữ cái trong câu văn.
        prioritized_patterns = [
            r"(?:CORRECT\s+ANSWER|FINAL\s+ANSWER|THE\s+ANSWER\s+IS|ANSWER\s*[:\-])\s*\(?\s*([A-D])\s*\)?\b",
            r"(?:BEST\s+RESPONSE|BEST\s+ANSWER)\s*(?:IS)?\s*[:\-]?\s*\(?\s*([A-D])\s*\)?\b",
            r"(?:^|\n)\s*([A-D])\s*[\).]?\s*$",
        ]

        predicted_letter = "None"
        for pattern in prioritized_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                predicted_letter = matches[-1]
                return (1.0, predicted_letter) if predicted_letter == expected_letter else (0.0, predicted_letter)

        # Fallback cuối: chỉ nhận chữ cái đơn lẻ ở gần cuối câu trả lời.
        tail = text_upper[-120:]
        tail_match = re.search(r"(?:^|\n)\s*([A-D])\s*[\).]?\s*$", tail)
        if tail_match:
            predicted_letter = tail_match.group(1)
            return (1.0, predicted_letter) if predicted_letter == expected_letter else (0.0, predicted_letter)

        return 0.0, "None"

    def calculate_p_norm(self, logprobs: List[float]) -> float:
        """
        Tính xác suất kết hợp (Joint Probability) của cả câu trả lời Y.
        Theo quy tắc chuỗi (Chain Rule) của Auto-regressive LLM:
        P(Y) = P(y_1) * P(y_2 | y_1) * ... * P(y_N | y_<N)
        Do cộng log() tương đương với nhân xác suất: P(Y) = exp(sum(logprobs))
        """
        if not logprobs:
            return 0.0
        # Cộng dồn log probability (tương đương nhân các xác suất có điều kiện)
        sum_log = float(np.sum(logprobs))
        # Khôi phục thành xác suất thực [0, 1]
        p_joint = math.exp(sum_log)
        return p_joint

    def calculate_ces(self, p_clean: float, p_perturbed: float) -> float:
        """
        Tính Causal Effect Score (CES).
        CES = P_norm(Y | do(G_clean)) - P_norm(Y | do(G_perturbed))
        CES cao chứng tỏ thay đổi trên Graph tác động mạnh đến mô hình (Nó nghe lời Graph).
        CES gần 0 chứng tỏ thay đổi không có hiệu ứng gì (Nó học vẹt).
        """
        return p_clean - p_perturbed

    def diagnose(
        self, question_id: str, responses: Dict[PerturbationType, LLMResponseInfo]
    ) -> CausalDiagnosisReport:
        """
        Chẩn đoán chi tiết sự phụ thuộc (Faithfulness vs Shortcuts) của mô hình bằng LLM-as-a-Judge Score.
        Vì Open-source LLM thường qua API không nhả Logprobs được như OpenAI, 
        nên ta thay P_norm bằng chính `is_correct` (điểm float 0.0 -> 1.0 từ Judge).
        CES = Điểm_CLEAN - Điểm_PERTURBED.
        """
        if PerturbationType.CLEAN not in responses:
            raise ValueError("Must provide CLEAN response as a baseline.")

        clean_res = responses[PerturbationType.CLEAN]

        # 1. Tổng quan: Baseline Accuracy trên Clean Context
        # Với MCQ score trả về là float {0.0, 1.0}
        clean_acc = float(clean_res.is_correct)
        p_clean = clean_acc

        # 2. Chi tiết: Tính CES cho mọi biến thể so với Clean
        ces_scores = {}
        for p_type, res in responses.items():
            if p_type != PerturbationType.CLEAN:
                ces_scores[p_type] = self.calculate_ces(p_clean, float(res.is_correct))

        # 3. Phân loại lỗi (Taxonomy Diagnosis)
        detected_shortcut = ShortcutType.UNKNOWN
        details = ""
        ces_threshold = 0.4  # Biến thiên 40% điểm số (giảm gần 1/2) mới được xem là chịu tác động

        if clean_acc < 0.5:
            details = "Baseline Failed: Mô hình trả lời sai ngay trên Clean Graph gốc."
            return CausalDiagnosisReport(
                question_id=question_id,
                clean_accuracy=clean_acc,
                causal_effect_scores=ces_scores,
                detected_shortcut=ShortcutType.UNKNOWN,
                details=details,
            )

        # Lấy dữ liệu phản hồi của Broken
        res_broken = responses.get(PerturbationType.BROKEN)
        ces_broken = ces_scores.get(PerturbationType.BROKEN, 0.0)

        # Kiểm tra 1: Parametric Memorization (Học vẹt)
        # Bị mất graph Evidence (Broken) nhưng lại tự trả lời ĐÚNG và không tụt điểm
        if res_broken and float(res_broken.is_correct) >= 0.5 and abs(ces_broken) < ces_threshold:
            detected_shortcut = ShortcutType.MEMORIZATION
            details = f"Parametric Memorization: Trả lời đúng dù Graph bị hỏng (CES_broken={ces_broken:.3f} quá thấp)."
            return CausalDiagnosisReport(
                question_id=question_id, 
                clean_accuracy=clean_acc, 
                causal_effect_scores=ces_scores, 
                detected_shortcut=detected_shortcut, 
                details=details
            )

        # Kiểm tra 2: Bị dẫn dụ bởi các bẫy nhiễu (Distractor Analysis)
        distractor_types = [
            PerturbationType.TOPOLOGICAL,
            PerturbationType.TYPE_MATCHING,
            PerturbationType.SWAPPING,
        ]
        max_distractor_ces = -1.0
        worst_trap = None

        for d_type in distractor_types:
            if d_type in responses and float(responses[d_type].is_correct) < 0.5:
                if ces_scores[d_type] > max_distractor_ces:
                    max_distractor_ces = ces_scores[d_type]
                    worst_trap = d_type

        # Nếu có trap nào kéo điểm mô hình xuống > 40%
        if worst_trap and max_distractor_ces > ces_threshold:
            if worst_trap == PerturbationType.TOPOLOGICAL:
                detected_shortcut = ShortcutType.TOPOLOGICAL_BIAS
                details = f"Topological Bias: Bị đánh lừa chọn nhầm node Hub phổ biến (CES_topo={max_distractor_ces:.3f})."
            elif worst_trap == PerturbationType.TYPE_MATCHING:
                detected_shortcut = ShortcutType.TYPE_SHORTCUT
                details = f"Type-matching Shortcut: Chọn nhầm thực thể cùng type thay vì đọc quan hệ logic (CES_type={max_distractor_ces:.3f})."
            elif worst_trap == PerturbationType.SWAPPING:
                detected_shortcut = ShortcutType.PRIOR_OVER_CONTEXT
                details = f"Prior-over-Context: Không tin vào sự nhiễu loạn của Graph mà vẫn tin vào Prior memory (CES_swap={max_distractor_ces:.3f})."

            return CausalDiagnosisReport(
                question_id=question_id, 
                clean_accuracy=clean_acc, 
                causal_effect_scores=ces_scores, 
                detected_shortcut=detected_shortcut, 
                details=details
            )

        # Kiểm tra 3: Causal Reasoning thực thụ
        # Không học vẹt (rớt điểm ở Broken - Broken trả lời 'None') + Không bị lừa (Vượt qua 3 bẫy nhiễu)
        detected_shortcut = ShortcutType.FAITHFUL
        details = (
            "Faithful Reasoning: Trả lời 'None' khi thiếu dữ liệu, chọn đúng ABCD trong mọi bẫy."
        )

        return CausalDiagnosisReport(
            question_id=question_id,
            clean_accuracy=clean_acc,
            causal_effect_scores=ces_scores,
            detected_shortcut=detected_shortcut,
            details=details,
        )
