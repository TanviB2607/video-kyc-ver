import cv2
import numpy as np
import insightface
from numpy.linalg import norm

class FaceMatchService:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0)

    def _get_embedding(self, img):
        faces = self.app.get(img)
        if not faces:
            return None
        return faces[0].embedding

    def compare_faces(self, id_face_img, video_frames):
        id_img = cv2.imread(id_face_img)
        emb_id = self._get_embedding(id_img)
        if emb_id is None:
            return 0.0

        scores = []
        for frame in video_frames[:10]:
            emb_vid = self._get_embedding(frame)
            if emb_vid is None:
                continue

            score = np.dot(emb_id, emb_vid) / (norm(emb_id) * norm(emb_vid))
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.0
