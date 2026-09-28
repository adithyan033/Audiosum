import asyncio

from speech_to_text import speech_to_text
from summarizer import summarize_transcript
from text_to_audio import text_to_audio


INPUT_AUDIO = "input.mp3"
SUMMARY_FILE = "summary.txt"
OUTPUT_AUDIO = "summarized_audio.mp3"


async def main():

    print("\n" + "=" * 60)
    print("          AUDIO SUMMARIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. SPEECH TO TEXT
    # --------------------------------------------------------

    print("\n[1/3] Converting speech to text...")

    transcript = speech_to_text(INPUT_AUDIO)

    if not transcript:
        print("\nERROR: Transcription failed.")
        return

    print("      ✓ Transcription completed")


    # --------------------------------------------------------
    # 2. TEXT SUMMARIZATION
    # --------------------------------------------------------

    print("\n[2/3] Generating summary...")

    summary = summarize_transcript(transcript)

    if not summary:
        print("\nERROR: Summarization failed.")
        return

    print("      ✓ Summary generated")


    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(summary)
    print("      ✓ Summary generated")
    print(f"      ✓ Summary saved: {SUMMARY_FILE}")


    # --------------------------------------------------------
    # 3. TEXT TO SPEECH
    # --------------------------------------------------------

    print("\n[3/3] Converting summary to audio...")

    await text_to_audio(
        summary,
        OUTPUT_AUDIO
    )

    print("      ✓ Audio generated")


    # --------------------------------------------------------
    # COMPLETED
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("              COMPLETED")
    print("=" * 60)

    print(f"\nSummary : {SUMMARY_FILE}")
    print(f"Audio   : {OUTPUT_AUDIO}")

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())