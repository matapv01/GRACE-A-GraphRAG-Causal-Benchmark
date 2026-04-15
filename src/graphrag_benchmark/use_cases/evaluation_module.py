import math
from typing import Dict, List
import numpy as np
from graphrag_benchmark.domain.models import (
    PerturbationType,
    ShortcutType,
    CausalDiagnosisReport,
    LLMResponseInfo,
)


class EvaluationModule:
    """
    Evaluate LLM faithfully against causal shortcuts.
    Từ tổng quan (Accuracy) đến bóc tách chi tiết (Token Logprobs -> P_norm -> CES).
    """

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
        Chẩn đoán chi tiết sự phụ thuộc (Faithfulness vs Shortcuts) của mô hình.
        """
        if PerturbationType.CLEAN not in responses:
            raise ValueError("Must provide CLEAN response as a baseline.")

        clean_res = responses[PerturbationType.CLEAN]

        # 1. Tổng quan: Baseline Accuracy trên Clean Context
        clean_acc = clean_res.exact_match
        p_clean = clean_res.p_norm

        # 2. Chi tiết: Tính CES cho mọi biến thể so với Clean
        ces_scores = {}
        for p_type, res in responses.items():
            if p_type != PerturbationType.CLEAN:
                ces_scores[p_type] = self.calculate_ces(p_clean, res.p_norm)

        # 3. Phân loại lỗi (Taxonomy Diagnosis)
        detected_shortcut = ShortcutType.UNKNOWN
        details = ""
        ces_threshold = 0.1  # Biến thiên 10% xác suất

        if not clean_acc:
            details = "Baseline Failed: Mô hình trả lời sai ngay cả trên Clean Graph."
            return CausalDiagnosisReport(
                question_id=question_id,
                clean_accuracy=clean_acc,
                causal_effect_scores=ces_scores,
                detected_shortcut=ShortcutType.UNKNOWN,
                details=details,
            )

        # Lấy dữ liệu phản hồi của từng bẫy (fallback nếu thiếu)
        res_broken = responses.get(PerturbationType.BROKEN)
        ces_broken = ces_scores.get(PerturbationType.BROKEN, 0.0)

        # Kiểm tra 1: Parametric Memorization (Học vẹt)
        # Nếu xóa chứng cứ (Broken) mà vẫn trả lời đúng và tự tin không giảm quá nhiều
        if res_broken and res_broken.exact_match and abs(ces_broken) < ces_threshold:
            detected_shortcut = ShortcutType.MEMORIZATION
            details = f"Parametric Memorization: Trả lời đúng dù thiếu Graph Evidence (CES_broken={ces_broken:.3f} quá thấp)."
            return CausalDiagnosisReport(
                question_id, clean_acc, ces_scores, detected_shortcut, details
            )

        # Kiểm tra 2: Bị dẫn dụ bởi các bẫy (Distractor Analysis)
        # Tính xem bẫy nào làm mô hình trả lời sai và làm sụt giảm xác suất P_norm lớn nhất
        distractor_types = [
            PerturbationType.TOPOLOGICAL,
            PerturbationType.TYPE_MATCHING,
            PerturbationType.SWAPPING,
        ]
        max_distractor_ces = -1.0
        worst_trap = None

        for d_type in distractor_types:
            if d_type in responses and not responses[d_type].exact_match:
                if ces_scores[d_type] > max_distractor_ces:
                    max_distractor_ces = ces_scores[d_type]
                    worst_trap = d_type

        if worst_trap and max_distractor_ces > ces_threshold:
            if worst_trap == PerturbationType.TOPOLOGICAL:
                detected_shortcut = ShortcutType.TOPOLOGICAL_BIAS
                details = f"Topological Bias: Bị đánh lừa bởi Hub Node (CES_topo={max_distractor_ces:.3f})."
            elif worst_trap == PerturbationType.TYPE_MATCHING:
                detected_shortcut = ShortcutType.TYPE_SHORTCUT
                details = f"Type-matching Shortcut: Chọn nhầm thực thể cùng type với Answer (CES_type={max_distractor_ces:.3f})."
            elif worst_trap == PerturbationType.SWAPPING:
                detected_shortcut = ShortcutType.PRIOR_OVER_CONTEXT
                details = f"Prior-over-Context: Không chú ý sự thay đổi logic hiển nhiên trên Graph (CES_swap={max_distractor_ces:.3f})."

            return CausalDiagnosisReport(
                question_id, clean_acc, ces_scores, detected_shortcut, details
            )

        # Kiểm tra 3: Faithful Reasoning (Causal Reasoning thực thụ)
        # Mô hình sập khi bị gãy (Broken), và VƯỢT QUA (exact_match=True) các bẫy nhiễu
        detected_shortcut = ShortcutType.FAITHFUL
        details = (
            "Faithful Reasoning: Vượt qua mọi bẫy, LLM dựa sát vào Graph để suy luận."
        )

        return CausalDiagnosisReport(
            question_id=question_id,
            clean_accuracy=clean_acc,
            causal_effect_scores=ces_scores,
            detected_shortcut=detected_shortcut,
            details=details,
        )
