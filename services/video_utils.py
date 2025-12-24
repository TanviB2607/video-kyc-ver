import cv2

def extract_frames(video_path, every_n_frames=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames
