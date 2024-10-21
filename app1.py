from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Muat model dari file H5
model = load_model('model/retinal_classification_model.h5')

# Fungsi untuk memproses gambar
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Ubah ukuran gambar
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    image = np.array(image) / 255.0  # Normalisasi gambar
    return image

# Route untuk halaman utama '/'
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('result.html', result='Error: No image uploaded.')

    file = request.files['image']
    if file.filename == '':
        return render_template('result.html', result='Error: No image selected.')

    try:
        image = Image.open(file)
        processed_image = preprocess_image(image, target_size=(150, 150))  # Sesuaikan dengan ukuran yang digunakan oleh model
        prediction = model.predict(processed_image)
        result = 'Retinal' if prediction[0] > 0.5 else 'Non-Retinal'
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('result.html', result=f'Error: {str(e)}')

# Jalankan Flask dengan debug mode
if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Ganti port jika diperlukan
