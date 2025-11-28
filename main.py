from fastapi import FastAPI, File, UploadFile
from face_engine import FaceEngine

app = FastAPI(
    title="Face Recognition API",
    description="InsightFace + SVM + KNN",
    version="1.0.0"
)

engine = FaceEngine()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = engine.read_image(img_bytes)

    emb = engine.get_embedding(img)
    if emb is None:
        return {"error": "No face detected"}

    result = engine.classify(emb)
    return result

@app.get("/")
def root():
    return {"msg": "Face API running!"}
