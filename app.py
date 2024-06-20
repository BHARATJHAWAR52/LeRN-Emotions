from flask import Flask, request, jsonify, render_template
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Load your saved emotion recognition model
model = load_model('D:\\RAVDESS Emotional speech audio\\ravdess_emotion_model_cnn.h5')

# Load label encoder and other preprocessing objects if necessary
label_encoder = LabelEncoder()

# Function to extract features from an audio file (similar to previous examples)
def extract_features(audio_path):
    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        
        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
    

# Function to preprocess audio file for prediction
def preprocess_audio(audio_path):
    # Extract features from the audio file
    features = extract_features(audio_path)
    
    if features is not None:
        # Normalize the features (assuming you used StandardScaler during training)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features.reshape(1, -1))
        
        # Reshape the input to match the model's expected input shape
        input_data = normalized_features.reshape(1, normalized_features.shape[1], 1, 1)
        
        return input_data
    else:
        return None
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Save the uploaded audio file
    audio_path = 'uploaded_audio.wav'
    file.save(audio_path)
    
    # Preprocess the audio file
    input_data = preprocess_audio(audio_path)
    
    if input_data is not None:
        # Make predictions using the loaded model
        predictions = model.predict(input_data)
        
        # Decode the predictions
        emotion_labels = ['calm', 'neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        predicted_label = emotion_labels[np.argmax(predictions)]
        # predicted_label=label_encoder.inverse_transform(np.argmax(predictions))
        
        return jsonify({'predicted_label': predicted_label})
    else:
        return jsonify({'error': 'Failed to process audio'})

if __name__ == '__main__':
    app.run(debug=True)

