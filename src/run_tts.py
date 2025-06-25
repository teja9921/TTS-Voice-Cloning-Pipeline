import torch
from TTS.api import TTS
import os
import soundfile as sf

# NOTE: This script is for inference using a pre-trained model or a fine-tuned local model.
# To use a fine-tuned model, change the model_path and config_path arguments in the TTS() call.

# --- CONFIGURATION ---
# Define relative paths for a modular structure
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) # This goes up one level from src to the main project directory
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Path to the reference audio file for voice cloning
REFERENCE_AUDIO = os.path.join(DATA_DIR, "reference_voice.wav") 

# Text to be synthesized
TEXT_TO_SPEAK = "Hello, this is a test of a custom-trained voice model. The quality and naturalness should be very high."

# Path for the output audio file
OUTPUT_AUDIO = os.path.join(OUTPUT_DIR, "output_clone.wav")

# --- MAIN SCRIPT ---
def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Using device: {device}")

    # Check if reference file exists
    if not os.path.exists(REFERENCE_AUDIO):
        print(f"Error: Reference audio file not found at '{REFERENCE_AUDIO}'")
        print("Please add a .wav file to the 'data' directory and name it 'reference_voice.wav'.")
        return

    print("--> Loading the TTS model...")
    # To use your fine-tuned model, you would use a line like this:
    # tts = TTS(model_path="path/to/your/best_model.pth", config_path="path/to/your/config.json").to(device)
    tts = TTS("tts_models/en/vctk/vits", progress_bar=True).to(device)

    print("--> Model loaded. Starting speech synthesis...")
    try:
        tts.tts_to_file(
            text=TEXT_TO_SPEAK,
            speaker_wav=REFERENCE_AUDIO,
            language="en",
            file_path=OUTPUT_AUDIO
        )
        print(f"--> Speech synthesis complete! Check for '{OUTPUT_AUDIO}' in your 'output' folder.")
    except Exception as e:
        print(f"An error occurred during synthesis: {e}")

if __name__ == '__main__':
    main()
