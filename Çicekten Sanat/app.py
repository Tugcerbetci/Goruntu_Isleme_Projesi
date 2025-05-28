from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
# Stil resimlerini internetten indirmek için
import requests
import shutil
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Klasör sabitleri
UPLOAD_FOLDER = 'static/uploads'      # Kullanıcının yüklediği fotoğraf buraya kaydedilecek
RESULT_FOLDER = 'static/results'      # Stil uygulanmış sonuç görüntüsü buraya kaydedilecek
STYLES_FOLDER = 'static/styles'       # Stil resimleri (örneğin Monet, Van Gogh) bu klasörde

# Yüklenebilecek dosya uzantıları
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Gerekli klasörleri oluştur (eğer yoksa)
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, STYLES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Dosya uzantısının izin verilen formatlarda olup olmadığını kontrol eder
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Stil resimlerinin indirildiğinden emin olur (yoksa indirir)
def ensure_style_images():
    style_images = {
        "monet": "https://www.istanbulsanatevi.com/wp-content/uploads/2022/04/claude-monet-aycicekleri.jpg",
        "vangogh": "https://pingpong.university/wp-content/uploads/2024/05/IB_API_P_5971661_6adf3d95-8847-41d6-bb6b-4c6512ae8ea9.jpg",
        "picasso": "https://i.pinimg.com/236x/d8/68/81/d86881fc4f7302d69584be58aac7c78a.jpg"
    }

    for style_name, url in style_images.items():
        style_path = os.path.join(STYLES_FOLDER, f"{style_name}.jpg")
        if os.path.exists(style_path):
            continue

        try:
            logger.info(f"Downloading {style_name} style...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(style_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            logger.info(f"{style_name} saved to {style_path}")
        except Exception as e:
            logger.error(f"Hata: {style_name} indirilemedi. {str(e)}")
            create_placeholder_image(style_path)

        
# Hata durumunda boş (gri renkli) bir görsel oluşturur
def create_placeholder_image(path):
    img = Image.new('RGB', (300, 300), color=(200, 200, 200))
    img.save(path, 'JPEG')
    logger.info(f"Placeholder created at {path}")


# Yedek placeholder görselinin var olup olmadığını kontrol eder, yoksa oluşturur
def ensure_placeholder_exists():
    placeholder_path = os.path.join(STYLES_FOLDER, "placeholder.jpg")
    if not os.path.exists(placeholder_path):
        create_placeholder_image(placeholder_path)

# TensorFlow Hub'dan stil transfer modeli yükleniyor
def load_model():
    try:
        return hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    #Bu model, TensorFlow Hub üzerinde barındırılan, Google Magenta ekibi tarafından önceden eğitilmiş bir "arbitrary image stylization" modelidir. 
    except Exception as e:
        logger.error(f"Model yükleme hatası: {str(e)}")
        return None

# Model bir kez yükleniyor ve bellekte tutuluyor
MODEL = load_model()

# Görseli yükleyip yeniden boyutlandırır ve modele uygun tensöre çevirir
def preprocess_image(image_path, max_dim=512):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Stil transfer işlemi burada yapılır
def style_transfer(content_path, style_path):
    try:
        if not os.path.exists(style_path):
            logger.error(f"Stil dosyası yok: {style_path}")
            return None

        content_image = preprocess_image(content_path)
        style_image = preprocess_image(style_path)

        if MODEL is None:
            logger.error("Model yüklenemedi.")
            return None

          # Stil transferi uygulanıyor
        stylized_image = MODEL(content_image, style_image)[0]
        stylized_image = tf.image.convert_image_dtype(stylized_image, tf.uint8)
        stylized_image = tf.squeeze(stylized_image).numpy()

        return stylized_image
    except Exception as e:
        logger.error(f"Stil transferi hatası: {str(e)}")
        return None


  # Ana sayfa (index) route
@app.route('/')
def index():
    ensure_style_images()         # Stil görselleri indiriliyor (varsa atlanır)
    ensure_placeholder_exists()   # Placeholder yedeği kontrol edilir
    styles = [                    # HTML tarafında gösterilecek stil seçenekleri
        {"name": "Monet", "filename": "monet.jpg"},
        {"name": "Van Gogh", "filename": "vangogh.jpg"},
        {"name": "Picasso", "filename": "picasso.jpg"}
    ]
    return render_template('index.html', styles=styles)

# Kullanıcı resim yüklediğinde işleme route'u
@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    style_name = request.form.get('style', 'vangogh')

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        content_path = os.path.join(UPLOAD_FOLDER, "content.jpg")
        file.save(content_path)

       # Stil dosyasının yolu
        style_path = os.path.join(STYLES_FOLDER, f"{style_name}.jpg")
        if not os.path.exists(style_path):
            style_path = os.path.join(STYLES_FOLDER, "placeholder.jpg")

        # Stil transferi uygulanıyor
        result_image = style_transfer(content_path, style_path)

        if result_image is not None:
            result_path = os.path.join(RESULT_FOLDER, "stylized.jpg")
            Image.fromarray(result_image).save(result_path)  # Görsel kaydediliyor

            # Başarı yanıtı
            return jsonify({
                "success": True,
                "result_url": f"/{RESULT_FOLDER}/stylized.jpg"
            })
        else:
            return jsonify({"error": "Style transfer failed"})  # İşlem başarısız
    else:
        return jsonify({"error": "File type not allowed"})  # Dosya tipi uygun değil

# Sunucuyu başlatır
if __name__ == '__main__':
    print("Uygulama başlatılıyor... http://127.0.0.1:5000 adresinde çalışacak")
    app.run(debug=True)