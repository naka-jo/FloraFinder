import torch  # PyTorchのインポート
import torchvision.models as models  # torchvisionの事前訓練済みモデルを使用するためのインポート
import torchvision.transforms as transforms  # 画像の前処理用モジュールのインポート
from PIL import Image  # 画像処理ライブラリPILのインポート
import torch.nn as nn  # ニューラルネットワークモジュールのインポート

NUM_CANDIDATES = 1  # 上位2つの予測結果を表示する設定
NUM_CLASSES = 4 # 分類したいクラス数

# 画像の前処理（リサイズ、テンソルへの変換、正規化）の設定
transform = transforms.Compose([
    transforms.Resize([224, 224]),  # 画像を224x224にリサイズ
    transforms.ToTensor(),  # 画像をテンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 画像の正規化
])

# 最後の全結合層をNUM_CLASSES分類用に変更し、重みデータのロード
model = models.resnet152(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load('Weights/model2-2.pth'))  # 学習済みの重みをロード
model.eval()  # モデルを評価モードに設定

# クラス名のリスト例　自分のクラス名・クラス数に合わせて編集すること
class_names = [
    "イチョウ",
    "ソテツ-雄株",
    "ソテツ-雌株",
    "マツ"
]

# 画像を予測する関数
def predict_22(image_path):
    image = Image.open(image_path)  # 画像を開く
    tensor_image = transform(image).unsqueeze(0)  # 画像をテンソルに変換し、バッチ次元を追加
    outputs = model(tensor_image)  # モデルに画像を入力し、出力を取得
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 出力をソフトマックス関数で確率に変換
    top_classes = torch.topk(probabilities, NUM_CANDIDATES)[1]  # 上位の確率とクラスのインデックスを取得
    # クラス名と確率を組み合わせてリストに格納
    results = class_names[top_classes[0][0]]
    return results  # 予測結果のリストを返す

# デバッグ用のmain関数
if __name__ == "__main__":
    image_path = "./Dataset/test_data/sotetsu_o.jpeg"  # テストする画像のパスを指定
    predictions = predict_22(image_path)  # 画像を予測する関数を呼び出す
    print(predictions)