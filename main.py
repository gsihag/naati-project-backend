# ===============================================
# Cross-Lingual Conversation Evaluator Backend
# ===============================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import whisper
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import tempfile, os, random, uuid
from gtts import gTTS

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title="Conversation Evaluator API")

# Allow frontend (Bolt / local dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # replace with ["http://localhost:5173", "https://your-frontend.com"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load models once
# -----------------------------
print("Loading Whisper ASR model...")
whisper_model = whisper.load_model("base")
print("Loading LaBSE model...")
labse_model = SentenceTransformer("sentence-transformers/LaBSE")
print("Models loaded successfully.")

# -----------------------------
# Conversation state
# -----------------------------
conversation_scores = []
current_segments = []
segment_index = 0

# Example prompts
english_prompts = [
    "The weather today is sunny and perfect for a long walk outside.",
    "Please make sure to complete all your assignments before the deadline.",
    "He bought a new bicycle to ride around the city every weekend."
]

hindi_prompts = [
    "आज का मौसम बहुत अच्छा है और बाहर लंबी सैर के लिए उपयुक्त है।",
    "कृपया समय सीमा से पहले अपने सभी असाइनमेंट पूरा करें।",
    "उसने सप्ताहांत में शहर में घूमने के लिए नई साइकिल खरीदी।"
]

# -----------------------------
# Helper functions
# -----------------------------
def generate_embeddings(text: str):
    emb = labse_model.encode([text])
    return emb[0]

def calculate_similarity(text1, text2):
    emb1 = generate_embeddings(text1)
    emb2 = generate_embeddings(text2)
    sim = 1 - cosine(emb1, emb2)
    return max(0.0, min(1.0, float(sim)))

def transcribe_audio(file_path: str, language: str = None):
    if language:
        result = whisper_model.transcribe(file_path, language=language)
    else:
        result = whisper_model.transcribe(file_path)
    return result["text"].strip()

def build_conversation(num_segments=3):
    segments = []
    for i in range(num_segments):
        if i % 2 == 0:  # English
            text = random.choice(english_prompts)
            lang = "en"
        else:           # Hindi
            text = random.choice(hindi_prompts)
            lang = "hi"
        segments.append({
            "prompt_text": text,
            "prompt_language": lang
        })
    return segments

# -----------------------------
# API Routes
# -----------------------------

@app.post("/start_conversation")
async def start_conversation(num_segments: int = 3):
    """
    Initialize a new conversation with alternating English/Hindi prompts.
    """
    global conversation_scores, current_segments, segment_index
    conversation_scores = []
    current_segments = build_conversation(num_segments)
    segment_index = 0
    return {"message": "Conversation started", "total_segments": len(current_segments)}

@app.get("/get_segment")
async def get_segment():
    """
    Get the next conversation segment with TTS audio.
    """
    global segment_index
    if segment_index >= len(current_segments):
        return {"message": "No more segments"}
    
    segment = current_segments[segment_index]
    text = segment["prompt_text"]
    lang = segment["prompt_language"]

    # Generate audio file with gTTS
    temp_audio_path = os.path.join(tempfile.gettempdir(), f"segment_{uuid.uuid4()}.mp3")
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_audio_path)

    return {
        "segment_index": segment_index,
        "prompt_text": text,
        "prompt_language": lang,
        "audio_url": f"/audio/{os.path.basename(temp_audio_path)}"
    }

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """
    Serve generated audio files.
    """
    file_path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

@app.post("/evaluate_response")
async def evaluate_response(
    response_audio: UploadFile = File(...),
    prompt_text: str = Form(...),
    prompt_language: str = Form(...)
):
    """
    Evaluate user's response for the current segment.
    """
    global segment_index, conversation_scores

    # Save uploaded audio
    temp_audio = os.path.join(tempfile.gettempdir(), f"user_{uuid.uuid4()}.wav")
    with open(temp_audio, "wb") as f:
        f.write(await response_audio.read())

    # Determine expected response language
    expected_lang = "hi" if prompt_language == "en" else "en"

    # Transcribe response
    response_text = transcribe_audio(temp_audio, language=expected_lang)

    # Calculate similarity
    score = calculate_similarity(prompt_text, response_text)
    conversation_scores.append(score)

    # Advance to next segment
    segment_index += 1

    return {
        "segment_index": segment_index - 1,
        "prompt_text": prompt_text,
        "prompt_language": prompt_language,
        "response_text": response_text,
        "similarity_score": score
    }

@app.get("/finish_conversation")
async def finish_conversation():
    """
    Finalize conversation and return average + normalized score.
    """
    if not conversation_scores:
        raise HTTPException(status_code=400, detail="No responses recorded")
    
    avg_score = float(np.mean(conversation_scores))
    normalized_score = avg_score * 45

    return {
        "segment_scores": conversation_scores,
        "average_score": avg_score,
        "normalized_score": normalized_score
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": True}


# Add this at the bottom of main.py after the health_check function
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
