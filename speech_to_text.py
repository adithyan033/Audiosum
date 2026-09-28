import whisper
import torch


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text(audio_file):

    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # print("\nLoading Whisper...")
    # print("Device:", device)

    # if device == "cuda":
    #     print("GPU:", torch.cuda.get_device_name(0))

    # Load Whisper model
    model = whisper.load_model(
        "base",
        device=device
    )

    # print("\nWhisper loaded successfully.")

    # # Transcribe audio
    # print("\nTranscribing audio...")

    result = model.transcribe(
        audio_file,
        language="en"
    )

    # Get transcript
    text = result["text"].strip()

    # print("\n" + "=" * 70)
    # print("TRANSCRIPTION")
    # print("=" * 70)

    # print(text)

    # Return transcript to the next component
    return text


# ============================================================
# TEST SPEECH TO TEXT SEPARATELY
# ============================================================

if __name__ == "__main__":

    audio_file = "input.mp3"

    transcript = speech_to_text(
        audio_file
    )