from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np
import io

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Face API is running!"}

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))

        # 這裡之後放你的 InsightFace 辨識
        # result = face_model.predict(image)

        # 先回傳基本資訊
        return {
            "filename": file.filename,
            "format": image.format,
            "size": image.size,
            "mode": image.mode,
            "status": "image received"
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
