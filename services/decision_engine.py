class DecisionEngine:
    """
    Final KYC decision engine.
    Produces: APPROVED | REVIEW | REJECTED
    """

    def __init__(
        self,
        face_threshold=0.75,
        doc_threshold=0.80,
        doc_match_threshold=0.80
    ):
        self.face_threshold = face_threshold
        self.doc_threshold = doc_threshold
        self.doc_match_threshold = doc_match_threshold

    def decide(
        self,
        S_face: float,
        S_live_p: float,
        S_live_a: float,
        S_doc: float,
        S_docmatch: float
    ) -> str:
        """
        Parameters:
        S_face       : Face similarity score (0-1)
        S_live_p     : Passive liveness (0 or 1)
        S_live_a     : Active liveness (0 or 1)
        S_doc        : Document authenticity score (0-1)
        S_docmatch   : Document-extracted data match (0-1)
        """

        if S_live_p == 0 or S_live_a == 0:
            return "REJECTED"

        if S_face < 0.55:
            return "REJECTED"

        if S_doc < 0.60 or S_docmatch < 0.60:
            return "REJECTED"

        
        if (
            S_face >= self.face_threshold and
            S_doc >= self.doc_threshold and
            S_docmatch >= self.doc_match_threshold and
            S_live_p == 1 and
            S_live_a == 1
        ):
            return "APPROVED"

        return "REVIEW"
    
    def explain(self, scores: dict) -> dict:
        """
        Explain why a decision was made (audit-friendly).
        """
        return {
            "face_ok": scores.get("face", 0) >= self.face_threshold,
            "doc_ok": scores.get("doc", 0) >= self.doc_threshold,
            "doc_match_ok": scores.get("doc_match", 0) >= self.doc_match_threshold,
            "passive_liveness_ok": int(scores.get("passive_liveness", 0)) == 1,
            "active_liveness_ok": int(scores.get("active_liveness", 0)) == 1
        }
