import requests
from typing import List
from graphrag_benchmark.domain.models import Triple, ReasoningSubgraph, PerturbedSubgraph, PerturbationType

class SandboxWrapper:
    """
    Manager to seamlessly switch the sandbox Server modes for Evaluation.
    """
    def __init__(self, endpoint_url: str = "http://localhost:8000"):
        self.endpoint_url = endpoint_url
        self.admin_url = f"{self.endpoint_url}/admin/load_triples"
        
    def _send_to_server(self, triples: List[Triple]):
        payload = [t.model_dump() for t in triples]
        try:
            res = requests.post(self.admin_url, json=payload, timeout=5)
            res.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Could not reach Sandbox Server (FastAPI). Ensure it is running: {e}")

    def load_clean_mode(self, subgraph: ReasoningSubgraph):
        """
        Mode 1: Chạy mô hình đánh giá trên Database sạch (Clean Subgraph ban đầu chưa qua can thiệp).
        Sử dụng kết quả này làm baseline P_norm(Y | do(G_clean)).
        """
        all_triples = list(set(subgraph.gold_path.triples + subgraph.extra_triples))
        self._send_to_server(all_triples)
        
    def load_perturbed_mode(self, perturbed: PerturbedSubgraph):
        """
        Mode 2: Chạy mô hình đánh giá trên Database đã bị nhiễm độc (Perturbed Subgraph).
        Sử dụng kết quả này để tính toán độ thay đổi hành vi P_norm(Y | do(G_perturbed)).
        """
        self._send_to_server(perturbed.triples)
        
    def switch_mode(self, mode: PerturbationType, subgraph: ReasoningSubgraph, perturbed: PerturbedSubgraph = None):
        """
        Convenience function to switch dynamically between any mode.
        """
        if mode == PerturbationType.CLEAN:
            self.load_clean_mode(subgraph)
        else:
            if not perturbed:
                raise ValueError("Must provide the PerturbedSubgraph data to use mode: " + mode.value)
            self.load_perturbed_mode(perturbed)
