from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import glob
import subprocess
import json
import warnings
import torch
import shutil
import gc 
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from faster_whisper import WhisperModel
from pydub import AudioSegment
from huggingface_hub import login

# --- The Notebook-Safe PyTorch Fix ---
if not hasattr(torch, "_is_patched"):
    _original_load = torch.load
    def _unlocked_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _unlocked_load
    torch._is_patched = True  

try:
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
except AttributeError:
    pass

warnings.filterwarnings("ignore")

app = FastAPI()

class VideoRequest(BaseModel):
    video_filename: str

@app.post("/extract")
def run_extraction(request: VideoRequest):
    print(f" Service A activated! Processing: {request.video_filename}")
    
    # --- SHARED FOLDER PATHS ---
    data_dir = "/app/data"
    video_path = f"{data_dir}/{request.video_filename}"
    raw_audio_path = f"{data_dir}/raw_audio.wav"
    demucs_out_dir = f"{data_dir}/separated"
    final_json_path = f"{data_dir}/final_master_transcript.json"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video {video_path} not found in shared volume.")

    # ==========================================
    # 1. EXTRACT RAW AUDIO
    # ==========================================
    print("1️⃣ Extracting raw audio...")
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {raw_audio_path} -y -loglevel error")

    # ==========================================
    # 2. PURIFY THE AUDIO (Demucs)
    # ==========================================
    print("2️⃣ Isolating vocals to prevent AI hallucinations...")
    subprocess.run(
        ["python", "-m", "demucs.separate", "-n", "htdemucs", "--two-stems", "vocals", "-d", "cuda", "-o", demucs_out_dir, raw_audio_path], 
        check=True
    )

    clean_vocal_path = glob.glob(f"{demucs_out_dir}/htdemucs/raw_audio/vocals.*")[0]
    shutil.copy(clean_vocal_path, f"{data_dir}/clean_vocals.wav")

    # ==========================================
    # 3. DIARIZATION (Pyannote)
    # ==========================================
    print("3️⃣ Loading Pyannote model into memory...")
    login("HF_TOKEN") 
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
    
    pipeline.instantiate({
        "clustering": {"threshold": 0.60}
    })
    
    print("Listening to PURIFIED audio with tuned sensitivity...")
    raw_diarization = pipeline(clean_vocal_path, min_speakers=1, max_speakers=5)

    diarization = Annotation()
    for turn, _, speaker in raw_diarization.itertracks(yield_label=True):
        if turn.end - turn.start > 0.2:
            diarization[turn] = speaker

    # ==========================================
    # 4. REFERENCE EXTRACTION (Pydub)
    # ==========================================
    print("4️⃣ Building dynamic voice references...")
    clean_vocals = AudioSegment.from_file(clean_vocal_path)
    unique_speakers = list(set([speaker for _, _, speaker in raw_diarization.itertracks(yield_label=True)]))

    for target_speaker in unique_speakers:
        longest_duration = 0
        best_start = 0
        best_end = 0
        
        for turn, _, speaker in raw_diarization.itertracks(yield_label=True):
            if speaker == target_speaker:
                duration = turn.end - turn.start
                if duration > longest_duration:
                    longest_duration = duration
                    best_start = turn.start
                    best_end = turn.end
        
        start_ms = int(best_start * 1000)
        end_ms = int(best_end * 1000)
        
        if (end_ms - start_ms) > 6000:
            end_ms = start_ms + 6000
            
        ref_filename = f"{data_dir}/ref_{target_speaker}.wav"
        clean_vocals[start_ms:end_ms].export(ref_filename, format="wav")

    # ==========================================
    #  THE VRAM ASSASSINATION 1 (Kill Pyannote)
    # ==========================================
    print(" Wiping Pyannote from GPU memory to make room for Whisper...")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # 5. TRANSCRIPTION (Faster-Whisper)
    # ==========================================
    print("5️⃣ Loading Whisper base to transcribe audio...")
    whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
    segments, info = whisper_model.transcribe(
        video_path, 
        word_timestamps=True,
        condition_on_previous_text=False # This stops the infinite repeating bug!
    )

    # ==========================================
    # 6. THE MERGE LOGIC
    # ==========================================
    print("6️⃣ Merging words with your speaker labels...")
    final_pipeline_data = []

    # Give the code a memory. We will assume Speaker 00 starts if the very first word is cut off.
    last_known_speaker = "SPEAKER_00"

    for segment in segments:
        for word in segment.words:
            word_start = round(word.start, 2)
            word_end = round(word.end, 2)
            word_midpoint = (word_start + word_end) / 2.0
            
            current_speaker = "UNKNOWN"
            
            # Ask Pyannote who was speaking (adding a 0.2-second buffer for breaths)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if (turn.start - 0.2) <= word_midpoint <= (turn.end + 0.2):
                    current_speaker = speaker
                    last_known_speaker = speaker # Update our memory!
                    break 
                    
            # If Pyannote missed it, use our memory to fill in the blank!
            if current_speaker == "UNKNOWN":
                current_speaker = last_known_speaker
                
            word_data = {
                "word": word.word.strip(),
                "start": word_start,
                "end": word_end,
                "speaker": current_speaker
            }
            final_pipeline_data.append(word_data)
            
            # Preview the merge in real-time
            print(f"[{word_start}s -> {word_end}s] {current_speaker}: {word_data['word']}")

    # ==========================================
    #  THE VRAM ASSASSINATION 2 (Kill Whisper)
    # ==========================================
    print(" Wiping Whisper from GPU memory...")
    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # 7. SAVE MASTER HANDOFF FILE
    # ==========================================
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_pipeline_data, f, indent=4, ensure_ascii=False)

    print(f"\n Service A Complete! Every word is labeled. Saved {final_json_path}")
    return {"status": "success", "file_ready": "final_master_transcript.json"}