import numpy as np
import librosa
import tensorflow as tf
import gradio as gr

# Load the pre-trained model
model = tf.keras.models.load_model("emotion_recognition_model.h5")

# Define emotion labels (adjust according to your dataset)
EMOTIONS = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}

def preprocess_audio(file_path):
    # Extract MFCCs (same as your notebook's `extract_features`)
    signal, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    mfccs = mfccs.T  # Transpose to (time_steps, 13)
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)  # Normalize
    return mfccs

def predict_emotion(audio_file):
    # Preprocess audio
    mfccs = preprocess_audio(audio_file)
    
    # Pad/Truncate to fixed length (e.g., 228 time steps)
    max_length = 228
    if mfccs.shape[0] < max_length:
        pad_width = max_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_length, :]
    
    # Reshape for model input (no flattening, keeping (1, 228, 13) shape)
    mfccs_reshaped = mfccs.reshape(1, max_length, 13)
    
    # Predict
    pred = model.predict(mfccs_reshaped)
    emotion_idx = np.argmax(pred)
    return EMOTIONS[emotion_idx]

# Create Gradio interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Emotion Recognition from Speech",
    description="Upload an audio file (.wav) to predict the speaker's emotion."
)

# Launch the app
interface.launch()
