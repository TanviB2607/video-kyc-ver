from flask import Flask, request, jsonify, render_template
import os
from services.video_utils import extract_frames
from services.face_match_service import FaceMatchService
from services.liveness_service import LivenessService
from services.document_service import DocumentService
from services.decision_engine import DecisionEngine

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    # ---- Validation ----
    if "id_doc" not in request.files or "video" not in request.files:
        return jsonify({"error": "Missing files"}), 400

    id_doc = request.files["id_doc"]
    video = request.files["video"]

    if id_doc.filename == "" or video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # ---- Save Files ----
    id_path = os.path.join(app.config["UPLOAD_FOLDER"], id_doc.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)

    id_doc.save(id_path)
    video.save(video_path)

    # ---- Initialize Services ----
    face_service = FaceMatchService()
    live_service = LivenessService()
    doc_service = DocumentService()
    decision_engine = DecisionEngine()

    # ---- Process Video ----
    frames = extract_frames(video_path)

    if not frames:
        return jsonify({"error": "Could not extract frames from video"}), 400

    # ---- Extract Face from ID ----
    id_face = doc_service.extract_face_from_id(id_path)

    if id_face is None:
        return jsonify({
            "status": "REJECTED",
            "reason": "Face not detected in ID document"
        }), 200

    # ---- Compute Scores ----
    S_face = face_service.compare_faces(id_face, frames)
    S_live_p = live_service.passive_liveness(frames)
    S_live_a = live_service.active_liveness(frames)
    S_doc = 0.90  # placeholder / model score

    # ---- Decision ----
    decision = decision_engine.decide(
        S_face, S_live_p, S_live_a, S_doc
    )

    scores = {
        "face": S_face,
        "passive_liveness": S_live_p,
        "active_liveness": S_live_a,
        "doc": S_doc,
    }

    explanation = decision_engine.explain(scores)

    # ---- Response ----
    return jsonify({
        "status": decision,
        "scores": scores,
        "explanation": explanation
    })


if __name__ == "__main__":
    app.run(debug=True)
