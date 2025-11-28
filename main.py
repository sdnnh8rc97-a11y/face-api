
import base64
import io
import os
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# ----------------------------
# 1. 初始化 FastAPI
# ----------------------------
app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 2. 載入 InsightFace 模型
# ----------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# ----------------------------
# 3. 載入 SVM / KNN 模型
# ----------------------------
MODEL_DIR = "./model"

with open(f"{MODEL_DIR}/svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open(f"{MODEL_DIR}/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open(f"{MODEL_DIR}/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# ----------------------------
# 4. Base64 輸入格式
# ----------------------------
class ImageBase64(BaseModel):
    image_base64: str

# ----------------------------
# 5. 提取 embedding
# ----------------------------
def extract_embedding(image):
    faces = face_app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding


# ----------------------------
# 6. 辨識 API
# ----------------------------
@app.post("/predict")
async def predict(data: ImageBase64):

    try:
        # Base64 → numpy image
        img_bytes = base64.b64decode(data.image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        emb = extract_embedding(img)
        if emb is None:
            return JSONResponse({"success": False, "msg": "No face detected"})

        # -------- SVM 預測 --------
        svm_pred = svm_model.predict([emb])[0]
        svm_name = label_map.get(svm_pred, "Unknown")

        # -------- KNN 預測 --------
        knn_pred = knn_model.predict([emb])[0]
        knn_name = label_map.get(knn_pred, "Unknown")

        return {
            "success": True,
            "svm_pred": svm_name,
            "knn_pred": knn_name,
        }

    except Exception as e:
        return JSONResponse({"success": False, "msg": str(e)})


# ----------------------------
# 7. Cloud Run 啟動
# ----------------------------
@app.get("/")
async def home():
    return {"msg": "Face API is running!"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
