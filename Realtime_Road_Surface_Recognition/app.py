from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64

# 引入推論模組和參數
from inference_image import get_model, predict, draw_boxes, NUM_CLASSES, CHECKPOINT_PATH, DEVICE, THRESHOLD, LABELS

app = Flask(__name__)

# 初始化模型
model = get_model(NUM_CLASSES, CHECKPOINT_PATH, DEVICE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    # 從客戶端接收影像
    data = request.files['image'].read()
    image = Image.open(io.BytesIO(data)).convert('RGB')

    # 執行推論
    boxes, labels, scores = predict(model, image, DEVICE, THRESHOLD)
    result = draw_boxes(image.copy(), boxes, labels, scores, LABELS)

    # 編碼回傳
    buffered = io.BytesIO()
    result.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return jsonify({'image': img_str})

if __name__ == '__main__':
    # 監聽所有介面，5000 Port
    app.run(host='0.0.0.0', port=5000)