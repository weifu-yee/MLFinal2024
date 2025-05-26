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
    print("[Server] /predict hit")
    if 'image' not in request.files:
        print("[Server] missing 'image' field")
        return jsonify({'error': 'no image'}), 400

    data = request.files['image'].read()
    print(f"[Server] received image bytes: {len(data)}")

    image = Image.open(io.BytesIO(data)).convert('RGB')
    boxes, labels, scores = predict(model, image, DEVICE, THRESHOLD)
    result = draw_boxes(image.copy(), boxes, labels, scores, LABELS)

    buffered = io.BytesIO()
    result.save(buffered, format='JPEG')
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    print(f"[Server] sending back base64 length: {len(img_str)}")
    return jsonify({'image': img_str})

if __name__ == '__main__':
    # 啟用 HTTPS (自簽憑證)
    app.run(
        host='0.0.0.0', port=5000,
        ssl_context=('cert.pem', 'key.pem')
    )