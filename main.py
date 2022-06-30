import shutil
from uuid import uuid4 as _uuid4

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

# 启动App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def uuid():
    return _uuid4().hex


def temp_file_name(filename: str) -> str:
    return f"{uuid()}_{filename}"


def copy_file(target: str, file):
    with open(target, "wb") as buffer:
        shutil.copyfileobj(file, buffer)


@app.post("/img:add")
def img_add(file: UploadFile = File(...)):
    fname = f"data/gallery/{temp_file_name(file.filename)}.jpg"
    copy_file(fname, file.file)
    return


@app.post("/img:query")
def img_query(file: UploadFile = File(...)):
    fname = f"data/query/{temp_file_name(file.filename)}.jpg"
    copy_file(fname, file.file)
    return


# 启动服务
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8181)
