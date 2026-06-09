# Deep Learning Voice Authenticator

A deep-learning voice anti-spoofing system trained on the **ASVspoof 5** dataset. The goal of this project is to accurately classify audio samples as either **bonafide** (real human voice) or **spoofed** (synthetically generated or manipulated voice) to secure voice authentication systems against adversarial attacks.

## Tech Stack & Infrastructure

*   **Frameworks:** TensorFlow and Keras
*   **Environment:** Google Colab (Python 3)
*   **Hardware:** NVIDIA A100 GPU
*   **Storage:** Google Cloud Storage (GCS) is utilized for handling large-scale audio datasets and TFRecords.

## Data Pipeline

*   **Training Dataset:** ASVspoof 5
*   **Preprocessing:** Audio is loaded as FLAC/WAV files, resampled, and transformed into Log-Mel Spectrograms.
*   **Storage:** The preprocessed features and labels are serialized into highly efficient `TFRecord` files, allowing for fast, batched, and pre-fetched ingestion via the `tf.data.Dataset` API.

## Project Architecture

To rigorously compare different neural network architectures on the audio data, our experiments have been separated into independent notebooks:

### 1. GRU Model (`GRU_Model.ipynb`)
**The Temporal Approach:** The initial architecture uses a Bidirectional Gated Recurrent Unit (BiGRU). Audio files are converted into log-mel spectrograms, treating the audio as a time-series sequence. The BiGRU excels at capturing the temporal dependencies and long-term sequential patterns in human speech versus synthetic speech.

### 2. CNN Model (`CNN_Model.ipynb`)
**The Spatial Approach:** A Convolutional Neural Network (CNN) architecture. By treating the log-mel spectrograms as 2D "images," the CNN focuses on extracting spatial textures, localized frequency anomalies, and spectral artifacts that are often left behind by text-to-speech (TTS) or voice conversion algorithms.

### 3. Synthesized Model (`Synthesized_Model.ipynb`)
**The Ensemble Approach:** Recognizing that CNNs and RNNs catch different types of spoofing artifacts, this model synthesizes the learnings from the standalone architectures. By combining the spatial feature extraction of the CNN with the temporal sequence modeling of the GRU, this approach aims to achieve state-of-the-art robustness.

## Real-World Generalization

To evaluate the generalization of the **Synthesized Model** on entirely unseen, real-world data, a custom inference dataset was constructed:

*   **Bonafide Data (53 Samples):** 53 original audio samples were recorded featuring various neutral, expressive, conversational, and phonetically rich sentences. All recordings were done using Audacity on a 2017 MacBook Pro.
*   **Spoofed Data (50 Samples):** The ElevenLabs platform (Starter plan, "Instant Voice Cloning" option using only 10 seconds of source audio) was utilized to clone one of the recorded voices. Following this, 50 synthetic samples were generated using the exact same sentences from the bonafide dataset.

**Zero-Shot Inference:** All 103 audio samples were passed through the combined Synthesized Model for inference. It is important to note that the model was **never trained** on this custom bonafide dataset prior to the test. This served as a strict evaluation of the model's ability to generalize to unseen speech audio in the wild.

**Results:** The Synthesized Model achieved an **84.5% accuracy** on this custom dataset.

## Authors

- **Ryan Ong** - Co-Creator - [LinkedIn](https://www.linkedin.com/in/ryan-ong-94b599289)
- **Joshua Thomas** - Co-Creator - [LinkedIn](https://www.linkedin.com/in/joshua-thomas-bb2422255)