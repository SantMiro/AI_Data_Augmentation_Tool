import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import keras

def determine_imbalance(data):
    '''
    Determine number of smaples depending on proportion of imbalance
    '''
    zero_class = data[data['Class']==0].count().iloc[-1]
    one_class = data[data['Class']==1].count().iloc[-1]
    return round((zero_class - one_class)/2)

# Assuming your GAN model is loaded and can generate synthetic data
def generate_synthetic_data(gan_model, num_samples, noise_dim):
    '''
    Generate synthetic data from the pre-trained generator from the GAN model.
    '''
    noise = np.random.normal(0, 1, (num_samples, noise_dim))
    synthetic_data = gan_model.predict(noise)
    return synthetic_data

def clip_negative_values(X, feature_indices):
    """ 
    Clip negative values to zero for the specified features.
    """
    X[:, feature_indices] = np.clip(X[:, feature_indices], a_min=50, a_max=None)
    return X

def concatenate_data(original_data, generated_data):
    '''
    Concatenate original dataset with the generated one.
    '''
    return pd.concat([original_data, generated_data])

def separate_features_and_target(data, target_columns):
    '''
    Separation of features and targets.
    '''
    X = data.drop(target_columns, axis=1)
    y = data[target_columns[-1]]
    return X, y

def standardize_features(X,reverse = False, scaler = None):
    '''
    Scale and reverse sacling of features.
    '''
    if not reverse:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided to inverse transform.")
        return scaler.inverse_transform(X)
    
def apply_smote(X, y, sampling_strategy='auto', random_state=42):
    '''
    Apply SMOTE techniques to guarantee balance in the dataset.
    '''
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    return smote.fit_resample(X, y)

def add_noise(X, noise_factor=0.001):
    '''
    Add noise to the dataset.
    '''
    noise = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    return X + noise

def scale_features(X, scale_factor=0.02):
    '''
    Further scaling of featues.
    '''
    scale = 1 + scale_factor * np.random.uniform(low=-1, high=1, size=X.shape)
    return X * scale

def data_augmentation_pipeline(data, target_columns, gan_model, noise_dim):
    '''
    Full Data Augmentation Pipeline
    '''
    # Step 0: Determine num_samples
    num_samples = determine_imbalance(data)
    
    # Step 1: Generate synthetic data using GAN
    generated_data = generate_synthetic_data(gan_model, num_samples, noise_dim)
    generated_data = clip_negative_values(generated_data,-1)
    generated_df = pd.DataFrame(generated_data, columns=data.columns[1:-1])
    generated_df[target_columns[-1]] = 1
    
    # Step 2: Concatenate original data with GAN-generated data
    combined_data = concatenate_data(data, generated_df)
    
    # Step 3: Separate features and target
    X, y = separate_features_and_target(combined_data, target_columns)

    # Step 4: Standardize features
    X_scaled, scaler = standardize_features(X)
    
    # Step 5: Apply SMOTE
    #X_resampled, y_resampled = apply_smote(X_scaled, y)
    
    # Step 6: Add random noise
    X_noisy = add_noise(X_scaled)
    
    # Step 7: Apply scaling to features
    X_augmented = scale_features(X_noisy)

    # Step 8: Inverse rescalinf of features
    X_rescaled = standardize_features(X_augmented,reverse = True,scaler = scaler)
    X_rescaled = clip_negative_values(X_rescaled, -1)
    print(len(X_rescaled),len(y))
    # Step 9: Return augmented data as DataFrame
    augmented_data = pd.DataFrame(X_rescaled, columns=data.columns[1:-1])
    #augmented_data[target_columns[-1]] = y#_resampled
    
    return augmented_data
