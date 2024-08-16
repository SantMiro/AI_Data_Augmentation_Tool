import pandas as pd
import keras
import pipeline

data_path = '../data/imbalanced_90_10.csv'
data = pd.read_csv(data_path)

loaded_model = keras.saving.load_model("../notebooks/generator_model.keras")

target_columns = ['id','Class']

augmented_data = pipeline.data_augmentation_pipeline(data, target_columns, loaded_model, noise_dim=100)

print(augmented_data.describe())

augmented_data.to_csv('../data/augmented_data.csv', index=False)