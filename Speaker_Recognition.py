import numpy as np
import librosa
from sklearn.mixture import GaussianMixture

audio_file = "D:/Nirma/Sem 6/ML/spass/Hetansh's voice/hetansh_voice.wav"

# Function to extract MFCC features from audio file
def extract_mfcc(audio_file, n_mfcc=13):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs

# Function to train Gaussian Mixture Model (GMM) using EM algorithm
def train_gmm(features, num_components=8, covariance_type='diag', max_iter=100):
    gmm = GaussianMixture(n_components=num_components, covariance_type=covariance_type, max_iter=max_iter)
    gmm.fit(features)
    return gmm

# Function to recognize speaker using trained GMMs
def recognize_speaker(test_features, gmms):
    scores = [gmm.score(test_features.reshape(1, -1)) for gmm in gmms]
    predicted_speaker = np.argmax(scores)
    return predicted_speaker

    # Example usage
    if __name__ == "__main__":
        # Example training data: paths to audio files for different speakers
        speaker1_files = ["D:NirmaSem 6MLspasshetanshspeaker1_1.wav", "D:NirmaSem 6MLspasshetanshspeaker1_2.wav", "D:NirmaSem 6MLspasshetanshspeaker1_3.wav"]
        speaker2_files = ["D:NirmaSem 6MLspasshimangispeaker2_1.wav", "D:NirmaSem 6MLspasshimangispeaker2_2.wav", "D:NirmaSem 6MLspasshimangispeaker2_3.wav"]

        # Extract MFCC features for each speaker's audio
        speaker1_features = np.concatenate([extract_mfcc(file) for file in speaker1_files])
        speaker2_features = np.concatenate([extract_mfcc(file) for file in speaker2_files])

        # Train GMMs for each speaker
        num_components = 8  # Number of components in the GMM
        speaker1_gmm = train_gmm(speaker1_features, num_components=num_components)
        speaker2_gmm = train_gmm(speaker2_features, num_components=num_components)

        # Example testing data: path to an audio file to recognize
        test_features = extract_mfcc(audio_file)

        # Recognize the speaker
        gmms = [speaker1_gmm, speaker2_gmm]
        predicted_speaker_index = recognize_speaker(test_features, gmms)
        if predicted_speaker_index == 0:
            print("Predicted speaker is Speaker 1")
        else:
            print("Predicted speaker is Speaker 2")
