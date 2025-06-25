# High-Fidelity Voice Cloning via TTS Model Fine-Tuning

This project is an end-to-end pipeline for creating a high-fidelity, custom voice clone by fine-tuning a Text-to-Speech (TTS) model on a specific speaker's audio data.

## Project Overview

The goal was to move beyond simple zero-shot cloning and produce a professional-grade voice clone with high naturalness and consistency. This was achieved by fine-tuning a base VITS model on a custom, single-speaker dataset, resulting in a new model that is an expert at speaking in that specific voice.

## Project Structure

TTS-Voice-Cloning-Pipeline/
│
├── data/
│   └── reference_voice.wav
│
├── output/
│   └── .gitkeep
│
├── src/
│   └── run_tts.py
│
├── .gitignore
├── requirements.txt
└── README.md


## The Pipeline

The project was structured as a multi-step ML pipeline:

1.  **Data Collection:** A custom dataset of ~20 minutes of high-quality audio was recorded from a single speaker.
2.  **Data Preprocessing:** The raw audio was segmented into ~1,000 short (3-10 second) `.wav` clips. Scripts were used to normalize volume and remove silence, ensuring a clean and consistent dataset. A `metadata.csv` file was manually created to map each audio file to its exact text transcription.
3.  **Model Fine-Tuning:** The base VITS model was fine-tuned on the custom dataset using the Coqui TTS command-line tools. The training process was controlled by a detailed `config.json` file where hyperparameters like learning rate (`0.0001`) and batch size (`32`) were specified.
4.  **Inference:** A Python script (`src/run_tts.py`) was developed to use the new fine-tuned model to synthesize speech from new text.

## Technology Stack

* **Python 3.11**
* **PyTorch**
* **Coqui TTS Library**
* **eSpeak-NG** (Phonemizer Backend)

## Setup and Usage

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project also has system-level dependencies. You will need to install the Microsoft C++ Build Tools and eSpeak-NG.*

3.  **Prepare your data:**
    * Place your reference audio file inside the `data/` folder and ensure it is named `reference_voice.wav`.

4.  **Run Inference:**
    * Run the script from the main project directory:
        ```bash
        python src/run_tts.py
        ```
    * The output will be saved in the `output/` folder as `output_clone.wav`.

## Key Challenges & Learnings

* **Environment Complexity:** Successfully managed complex system-level dependencies (Python version, C++ compiler, eSpeak-NG), which is common for advanced AI libraries.
* **Data Sensitivity:** Learned that model output quality is extremely sensitive to the consistency and cleanliness of the training data. Iterating on the data preprocessing pipeline was the most critical factor for achieving a high-quality result.

## Future Work

The next logical step for this project is to integrate a model like **StyleTTS 2** to enable control over the emotion and prosody of the synthesized speech. This would allow the cloned voice to speak in different tones (e.g., happy, empathetic, professional), which is crucial for real-world applications in areas like healthcare AI.
