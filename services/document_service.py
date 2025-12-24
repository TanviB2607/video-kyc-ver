import cv2
import mediapipe as mp

class DocumentService:
    def __init__(self):
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def extract_face_from_id(self, img_path):
        img = cv2.imread(img_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.face_detector.process(rgb)

        if not results.detections:
            return None

        h, w, _ = img.shape
        bbox = results.detections[0].location_data.relative_bounding_box

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        return img[y1:y2, x1:x2]
