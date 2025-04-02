import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import librosa
import os

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Function to load and preprocess audio for YAMNet
def preprocess_audio_for_yamnet(file_path):
    # YAMNet expects 16 kHz mono audio
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    return audio, sr

# Test YAMNet on a sample file
base_dir = "/data/birdclef/birdclef-2025"
sample_file = os.path.join(base_dir, "train_soundscapes", os.listdir(os.path.join(base_dir, "train_soundscapes"))[0])

print(f"Testing YAMNet on: {sample_file}")
audio, sr = preprocess_audio_for_yamnet(sample_file)

# Run the model on the waveform
scores, embeddings, log_mel_spectrogram = yamnet_model(audio)

# Print the shape of the embeddings
print(f"Embeddings shape: {embeddings.shape}")
print(f"Number of segments: {embeddings.shape[0]}")
print(f"Embedding dimension: {embeddings.shape[1]}")