import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import pickle
import json

class FaceEngine:
    def __init__(self):
        print("⚙️ Initializing face engine...")

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # -----------------------------
        # Load SVM / KNN / label_map
        # -----------------------------
        with open("model/svm_model.pkl", "rb") as f:
            self.svm = pickle.load(f)

        with open("model/knn_model.pkl", "rb") as f:
            self.knn = pickle.load(f)

        with open("model/label_map.json", "r") as f:
            self.label_map = json.load(f)

    def read_image(self, file_bytes):
        img = np.frombuffer(file_bytes, np.uint8)
        return cv2.imdecode(img, cv2.IMREAD_COLOR)

    def get_embedding(self, img):
        faces = self.app.get(img)
        if len(faces) == 0:
            return None
        return faces[0].normed_embedding

    def classify(self, emb):
        # cosine similarity
        sims = self.svm.decision_function([emb])[0]  # SVM confidence
        svm_pred = self.svm.predict([emb])[0]

        # KNN confidence (inverse distance)
        dist, idx = self.knn.kneighbors([emb], n_neighbors=1)
        knn_conf = float(1 / (1 + dist[0][0]))
        knn_pred = self.knn.predict([emb])[0]

        return {
            "svm_pred": self.label_map[str(svm_pred)],
            "svm_conf": float(max(sims)),

            "knn_pred": self.label_map[str(knn_pred)],
            "knn_conf": knn_conf
        }
