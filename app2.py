from flask import Flask, request, jsonify, send_file
from google import genai
from dotenv import load_dotenv
import os
import wave
import contextlib
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

file_counter = 0  # To generate unique audio filenames

# Helper function to save audio blob to WAV
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

def save_audio_blob(blob, language="output"):
    global file_counter
    file_counter += 1
    fname = f"audio_{file_counter}.wav"
    with wave_file(fname) as wav:
        wav.writeframes(blob.data)
    return fname

def generate_audio(text, language):
    MODEL_ID = "gemini-2.5-flash-preview-tts"
    prompt = f"You are an expert medical translator and your pronunciation is suitable for medical contexts. Generate a text-to-speech audio in {language} for the following text: {text}."
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config={"response_modalities": ['Audio']}
    )
    blob = response.candidates[0].content.parts[0].inline_data
    fname = save_audio_blob(blob, language)
    return fname

@app.route("/speech_translate", methods=["POST"])
def speech_translate():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["file"]
        input_language = request.form.get("input_language")
        output_language = request.form.get("output_language")

        if not input_language or not output_language:
            return jsonify({"error": "Both input_language and output_language are required"}), 400

        # Save uploaded file temporarily
        temp_path = "temp_audio.mp3"
        file.save(temp_path)
        myfile = client.files.upload(file="temp_audio.mp3")

        # Speech-to-text + translation
        prompt = f"""
       Strictly transcribe the audio file in {input_language} exactly as spoken, without adding or removing any words.
       Then translate it exactly into {output_language}.
       Output only the  translated text.dont print/give the input/orignal text, without any extra text, explanations, or headings.
       """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, myfile]
        )
        translated_text = response.text
        print("translated_text:", translated_text)

        # Generate TTS audio for translated text
        audio_file = generate_audio(translated_text, output_language)

        # Clean up uploaded audio
        os.remove(temp_path)

        return jsonify({
            "input_language": input_language,
            "output_language": output_language,
            "translated_text": translated_text,
            "audio_file": audio_file  # frontend can fetch this file for playback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional endpoint to serve audio files
@app.route("/audio/<filename>")
def get_audio(filename):
    return send_file(filename, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)
