from flask import Flask, render_template, flash, request, session, send_file
from flask import render_template, redirect, url_for, request
import os

app = Flask(__name__)
app.config['DEBUG']
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('Prediction.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        import tensorflow as tf
        import numpy as np
        import cv2
        from keras.preprocessing import image
        file = request.files['file']
        file.save('static/upload/Test.jpg')
        fname = 'static/upload/Test.jpg'

        img1 = cv2.imread('static/upload/Test.jpg')

        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        noi = 'static/upload/noi.jpg'
        cv2.imwrite(noi, dst)

        import warnings
        warnings.filterwarnings('ignore')

        base_dir = 'Data/Train/'

        catgo = os.listdir(base_dir)
        print(catgo)
        classifierLoad = tf.keras.models.load_model('model.h5')
        test_image = image.load_img('static/upload/Test.jpg', target_size=(100, 100))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        print(result)
        ind = np.argmax(result)

        print(catgo[ind])
        out = ''

        if catgo[ind] == "lung_aca":
            out ='lung adenocarcinoma'
            pre = "Use Atovaquone-proguanil (Malarone) Quinine sulfate (Qualaquin) with doxycycline (Oracea, Vibramycin, others) Primaquine phosphate."
        elif catgo[ind] == "lung_n":
            out = 'lung benign'
            pre = "Nill"
        elif catgo[ind] == "lung_scc":
            out = 'lung squamous cell carcinoma'
            pre = " Use Atovaquone-proguanil (Malarone) Quinine sulfate (Qualaquin) with doxycycline (Oracea, Vibramycin, others) Primaquine phosphate."

        return render_template('Result.html', fer=pre, result=out, org=fname,noi=noi )


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
