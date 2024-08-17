# src/app.py
from flask import Flask, render_template, request,send_file
import pandas as pd
import keras
import pipeline
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/' 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/augment',methods=['POST'])
def augment_data():
    if request.method == 'POST':
        dataset = request.files['dataset']
        if dataset:
            dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
            dataset.save(dataset_path)

#data_path = '../data/imbalanced_90_10.csv'
            data = pd.read_csv(dataset_path)

            loaded_model = keras.saving.load_model("./model/generator_model.keras")

            target_columns = ['id','Class']

            augmented_data = pipeline.data_augmentation_pipeline(data, target_columns, loaded_model, noise_dim=100)

            # Generate a unique filename for the augmented data
            augmented_filename = f"augmented_data_{uuid.uuid4().hex}.csv"
            augmented_file_path = os.path.join(app.config['UPLOAD_FOLDER'], augmented_filename)

            augmented_data.to_csv(augmented_file_path, index=False)

            # Provide download link in the response
            return render_template('index.html', download_link=augmented_filename)
        
@app.route('/download/<filename>',methods = ['GET'])
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)




# print(augmented_data.describe())

# augmented_data.to_csv('../data/augmented_data.csv', index=False)