from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from inference_1 import predict_1  # 自作のpredictモジュールをインポート
from inference_21 import predict_21
from inference_22 import predict_22
from inference_31 import predict_31
from inference_32 import predict_32
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# 画像を保存するディレクトリを指定
UPLOAD_DIR = "Uploaded_images"
# 指定したディレクトリが存在しない場合、ディレクトリを作成
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    
app.mount("/static", StaticFiles(directory="Uploaded_images"), name="static")

@app.get("/")
def main(request: Request):
    # HTMLファイルを返す
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # アップロードされたファイルを指定のディレクトリに保存するパスを指定
    global input_image_path
    global output_image
    input_image_path = os.path.join(UPLOAD_DIR, file.filename)
    output_image = file.filename
    
    # アップロードされたファイルを指定したパスに保存
    with open(input_image_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # predict関数を呼び出して推論結果を取得
    result_text = predict_1(input_image_path)
    
    if result_text == "裸子植物":
        result_text = predict_22(input_image_path)
        final_result = "裸子植物" +" "+result_text
        text0 = "裸子植物とは？"
        text1 = "・・・種子が剥き出しになっている"
        text2 = "・・・種子植物の祖先"
        text3 = "スギ、モミ、ヒノキなど"
        
    if result_text == "被子植物":
        return templates.TemplateResponse("inference_21.html", 
                                            {"request": request, 
                                            "result_text": result_text})
    
    image_url = f"/static/{file.filename}"
    
    # 推論結果をHTML形式で返す
    return templates.TemplateResponse("result.html",  # 使用するHTMLファイル名を指定
                                        {"request": request,
                                        "image_url": image_url,
                                        "final_result": final_result,
                                        "text0": text0,
                                        "text1": text1,
                                        "text2": text2,
                                        "text3": text3})
    
@app.post("/predict21")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # アップロードされたファイルを指定のディレクトリに保存するパスを指定
    input_image_path_2 = os.path.join(UPLOAD_DIR, file.filename)
    
    # アップロードされたファイルを指定したパスに保存
    with open(input_image_path_2, "wb") as buffer:
        buffer.write(await file.read())
    
    # predict関数を呼び出して推論結果を取得
    result_text = predict_21(input_image_path_2)
    
    if result_text == "単子葉類":
        result_text = predict_31(input_image_path)
        final_result = "被子植物 単子葉類"
        text0 = "単子葉類とは？"
        text1 = "・・・子葉が1枚で、葉脈が平行"
        text2 = "・・・根はひげ状に伸びている"
        text3 = "トウモロコシ、ネギなど"
    else:
        result_text = predict_32(input_image_path)
        final_result = "被子植物 双子葉類"
        text1 = "双子葉類・・・子葉が2枚で、葉脈は網状"
        if "合弁花類" in result_text:
            text0 = "双子葉類 合弁花類とは？"
            text2 = "合弁花類・・・花弁がひとつに繋がっている"
            text3 = "キク、アサガオ、ナスなど"
        else:
            text0 = "双子葉類 離弁花類とは？"
            text2 = "離弁花類・・・花弁が1枚ずつ分離している"
            text3 = "サクラ、ウメ、エンドウなど"
    final_result += ""
    final_result += result_text
        
    
    image_url = f"/static/{output_image}"
        
    # 推論結果をHTML形式で返す
    return templates.TemplateResponse("result.html",  # 使用するHTMLファイル名を指定
                                        {"request": request,
                                        "image_url": image_url,
                                        "final_result": final_result,
                                        "text0": text0,
                                        "text1": text1,
                                        "text2": text2,
                                        "text3": text3})