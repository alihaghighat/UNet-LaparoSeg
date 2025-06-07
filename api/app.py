from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from model import predict_single_image  # فرض بر اینه که این تابع موجوده

app = FastAPI()

# 🚨 مهم‌ترین بخش برای فعال کردن CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # برای تست: همه مجاز، بعداً فقط دامنه خودتو بذار
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# فایل‌های خروجی تصویری رو هم سرو کن
app.mount("/output_single", StaticFiles(directory="output_single"), name="output")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # اجرای تابع پیش‌بینی
    report = predict_single_image(temp_filename, save_dir="output_single", alpha=0.5)

    basename = Path(temp_filename).stem
    overlay_path = f"output_single/{basename}_overlay.png"

    os.remove(temp_filename)

    return JSONResponse(content={
        "overlay_image_path": overlay_path,
        "report": report
    })
