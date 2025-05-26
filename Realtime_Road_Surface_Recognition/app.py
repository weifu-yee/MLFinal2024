from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

# 引入推論模組和參數
from inference_image import (
    get_model,
    predict,
    draw_boxes,
    NUM_CLASSES,
    CHECKPOINT_PATH,
    DEVICE,
    THRESHOLD,
    LABELS
)

app = Flask(__name__)
# 初始化模型（啟動時載入一次）
model = get_model(NUM_CLASSES, CHECKPOINT_PATH, DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ping', methods=['GET'])
def ping():
    print("[Server] /ping hit", flush=True)
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict_route():
    print("[Server] /predict hit", flush=True)

    # 僅支援 JSON (Base64 Data URL)
    if not request.is_json:
        print("[Server] Request Content-Type is not application/json", flush=True)
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    b64str = data.get('image')
    if not b64str:
        print("[Server] JSON 中缺少 'image' 欄位", flush=True)
        return jsonify({'error': 'no image field in JSON'}), 400

    # 拆除 Data URL 的 header
    try:
        header, b64 = b64str.split(',', 1)
    except ValueError:
        print("[Server] Data URL 格式錯誤", flush=True)
        return jsonify({'error': 'invalid image data URL'}), 400

    # Base64 解碼
    try:
        img_bytes = base64.b64decode(b64)
    except Exception as e:
        print(f"[Server] Base64 解碼失敗: {e}", flush=True)
        return jsonify({'error': 'cannot decode base64'}), 400

    # 以 PIL 開圖
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception as e:
        print(f"[Server] PIL 開圖失敗: {e}", flush=True)
        return jsonify({'error': 'cannot open image'}), 400

    # 模型推論
    print("[Server] Running inference...", flush=True)
    boxes, labels, scores = predict(model, image, DEVICE, THRESHOLD)

    # Log 出辨識出的 class（包含 background）
    detected = [LABELS[label] for label in labels]
    all_classes = ['background'] + detected
    print(f"[Server] Detected classes: {all_classes}", flush=True)

    # 繪製框框
    result = draw_boxes(image.copy(), boxes, labels, scores, LABELS, text_scale=3.0)

    # 將結果轉成 JPEG 再 Base64 編碼
    buffered = io.BytesIO()
    result.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(f"[Server] 回傳 base64 長度: {len(img_str)}", flush=True)

    return jsonify({'image': img_str, 'classes': detected})

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        ssl_context=('cert.pem', 'key.pem'),
        debug=False
    )
