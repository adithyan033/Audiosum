import asyncio
import edge_tts

async def text_to_speech(text):
    voice = "en-US-GuyNeural"
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("output.mp3")

# Read text from input file
with open("sharunabouttest.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Convert text to speech
asyncio.run(text_to_speech(text))