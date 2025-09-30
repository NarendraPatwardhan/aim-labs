import asyncio
import io
import logging
import sqlite3
import time
import uuid

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = "cpu"
torch.set_grad_enabled(False)  # Disable gradient tracking for inference

# Load Whisper model asynchronously
logger.info("Loading Whisper model...")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny"
).to(device)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Frontend origins allowed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Database initialization
conn = sqlite3.connect("voice_notes.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS voice_notes (
        id TEXT PRIMARY KEY,
        transcription TEXT,
        tags TEXT,
        audio BLOB,
        created_at REAL
    )
"""
)
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS drafts (
        draft_id TEXT PRIMARY KEY,
        transcription TEXT,
        tags TEXT,
        audio BLOB,
        created_at REAL
    )
"""
)
conn.commit()


async def transcribe_audio(audio_array, sampling_rate):
    """Transcribe audio using Whisper in a separate thread"""
    start_time = time.time()

    input_features = whisper_processor(
        audio_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(device)

    forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
        language="it", task="transcribe"
    )

    predicted_ids = await asyncio.to_thread(
        whisper_model.generate,
        input_features,
        max_new_tokens=256,
        forced_decoder_ids=forced_decoder_ids,
    )

    transcribed_text = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]

    duration = time.time() - start_time
    logger.info(f"ASR processing time: {duration:.2f} seconds")
    return transcribed_text


def generate_tags(text, max_tags=5):
    """Simple placeholder tag generation from transcription"""
    words = [w.strip().lower() for w in text.split() if len(w) > 3]
    unique_words = list(dict.fromkeys(words))  # remove duplicates
    return unique_words[:max_tags]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/voice-notes/draft")
async def create_draft(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".webm")):
        raise HTTPException(
            status_code=400, detail="Only .wav or .webm files are supported"
        )

    try:
        # Read the file content into memory
        file_content = await file.read()

        if file.filename.endswith(".webm"):
            # Convert .webm to PCM audio using pydub
            audio = AudioSegment.from_file(io.BytesIO(file_content), format="webm")
            # Convert to mono and set sample rate to 16000 Hz (Whisper-compatible)
            audio = audio.set_channels(1).set_frame_rate(16000)
            # Export to in-memory WAV buffer
            buffer = io.BytesIO()
            audio.export(buffer, format="wav")
            buffer.seek(0)
            # Read the converted audio
            audio_data, sample_rate = sf.read(buffer)
        else:
            # Handle .wav file directly
            audio_data, sample_rate = sf.read(io.BytesIO(file_content))

        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        transcription = await transcribe_audio(audio_data, sample_rate)
        tags = generate_tags(transcription)

        draft_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO drafts (draft_id, transcription, tags, audio, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                draft_id,
                transcription,
                ",".join(tags),
                audio_data.tobytes(),
                time.time(),
            ),
        )
        conn.commit()

        return {"draftId": draft_id, "transcription": transcription, "tags": tags}
    except Exception as e:
        logger.error(f"Error creating draft: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio file")


@app.put("/voice-notes/draft/{draft_id}")
async def update_draft(
    draft_id: str, transcription: str = None, tags: list[str] = Query(None)
):
    cursor.execute("SELECT * FROM drafts WHERE draft_id = ?", (draft_id,))
    draft = cursor.fetchone()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    new_transcription = transcription if transcription else draft[1]
    new_tags = ",".join(tags) if tags else draft[2]

    cursor.execute(
        "UPDATE drafts SET transcription = ?, tags = ? WHERE draft_id = ?",
        (new_transcription, new_tags, draft_id),
    )
    conn.commit()

    return {
        "draftId": draft_id,
        "transcription": new_transcription,
        "tags": new_tags.split(","),
    }


@app.post("/voice-notes")
async def finalize_draft(draftId: str = Query(...)):
    cursor.execute("SELECT * FROM drafts WHERE draft_id = ?", (draftId,))
    draft = cursor.fetchone()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    note_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO voice_notes (id, transcription, tags, audio, created_at) VALUES (?, ?, ?, ?, ?)",
        (note_id, draft[1], draft[2], draft[3], time.time()),
    )
    cursor.execute("DELETE FROM drafts WHERE draft_id = ?", (draftId,))
    conn.commit()

    return {
        "id": note_id,
        "transcription": draft[1],
        "tags": draft[2].split(","),
        "createdAt": time.time(),
    }


@app.get("/voice-notes")
async def list_notes(search: str = Query(None), tags: str = Query(None)):
    query = "SELECT id, transcription, tags, created_at FROM voice_notes"
    filters = []
    params = []

    if search:
        filters.append("transcription LIKE ?")
        params.append(f"%{search}%")
    if tags:
        tag_list = tags.split(",")
        filters.append(" OR ".join(["tags LIKE ?" for _ in tag_list]))
        params.extend([f"%{t}%" for t in tag_list])

    if filters:
        query += " WHERE " + " AND ".join(filters)

    query += " ORDER BY created_at DESC"

    cursor.execute(query, tuple(params))
    notes = cursor.fetchall()
    result = [
        {"id": n[0], "transcription": n[1], "tags": n[2].split(","), "createdAt": n[3]}
        for n in notes
    ]
    return result


@app.get("/voice-notes/{note_id}")
async def get_note(note_id: str):
    cursor.execute(
        "SELECT id, transcription, tags, created_at FROM voice_notes WHERE id = ?",
        (note_id,),
    )
    note = cursor.fetchone()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return {
        "id": note[0],
        "transcription": note[1],
        "tags": note[2].split(","),
        "createdAt": note[3],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
