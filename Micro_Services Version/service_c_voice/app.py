import gradio as gr
import requests
import os
import json
import glob
import subprocess
import warnings
import torch
import shutil
import gc 
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.silence import detect_nonsilent
from TTS.api import TTS
from TTS.utils.manage import ModelManager 

warnings.filterwarnings("ignore")

# ==========================================
#  PRE-DOWNLOAD DIRECTLY TO DISK ON BOOT
# ==========================================
print(" Checking/Downloading XTTS-v2 to disk (This may take a while)...")
ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
print(" XTTS-v2 Model is fully downloaded and securely cached on your hard drive!")

def process_pipeline(uploaded_video_path):
    # --- SHARED FOLDER ---
    data_dir = "/app/data"
    shared_video_path = f"{data_dir}/source_video.mp4"
    
    print(" Saving video to shared volume...")
    shutil.copy(uploaded_video_path, shared_video_path)
    
    # ==========================================
    #  THE ORCHESTRATOR
    # ==========================================
    print(" Pinging Service A (Extraction & Diarization)...")
    res_a = requests.post("http://dub_service_a:8001/extract", json={"video_filename": "source_video.mp4"}).json()
    transcript_name = res_a["file_ready"]
    
    print(" Pinging Service B (NLLB Translation)...")
    res_b = requests.post("http://dub_service_b:8002/translate", json={"transcript_filename": transcript_name}).json()
    translated_json_name = res_b["file_ready"]
    
    # ==========================================
    #  RUNNING NOTEBOOK 3 LOGIC (Voice & Mix)
    # ==========================================
    print("\n STARTING THE ADVANCED SYNC PIPELINE...\n")
    transcript_path = f"{data_dir}/{translated_json_name}"
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------------------------------------------
    #  Extract base audio for Voice References
    # ------------------------------------------
    print("1️⃣ Extracting original audio for voice cloning references...")
    base_audio_path = f"{data_dir}/base_audio.wav"
    os.system(f"ffmpeg -i {shared_video_path} -vn -acodec pcm_s16le -ar 22050 -ac 1 {base_audio_path} -y -loglevel error")
    base_audio = AudioSegment.from_wav(base_audio_path)

    # ------------------------------------------
    #  Automated Vocal Isolation & 6-Second Hunter
    # ------------------------------------------
    print("2️⃣ Fully Automated Vocal Isolation (Removing Music/SFX)...")
    subprocess.run(["demucs", "-d", "cuda", "--two-stems", "vocals", "-o", data_dir, base_audio_path], check=True)

    vocal_stem_path = f"{data_dir}/htdemucs/base_audio/vocals.wav"
    if not os.path.exists(vocal_stem_path):
        print(" Path adjustment: Searching for vocal stem...")
        vocal_stem_path = glob.glob(f"{data_dir}/*/base_audio/vocals.wav")[0]

    clean_vocals = AudioSegment.from_wav(vocal_stem_path)

    refs = {}
    for line in data:
        spk = line["speaker"]
        duration = line["end"] - line["start"]
        
        if spk not in refs:
            refs[spk] = {"start": line["start"], "end": line["end"], "duration": duration}
        else:
            current_best_diff = abs(refs[spk]["duration"] - 6.0)
            new_diff = abs(duration - 6.0)
            if new_diff < current_best_diff:
                refs[spk] = {"start": line["start"], "end": line["end"], "duration": duration}

    for spk, info in refs.items():
        start_ms = int(info["start"] * 1000)
        end_ms = int(info["end"] * 1000)
        ref_path = f"{data_dir}/ref_{spk}.wav"
        clean_vocals[start_ms:end_ms].export(ref_path, format="wav")
        print(f"    -> Extracted music-free reference for {spk} ({info['duration']:.2f}s)")

    # ------------------------------------------
    #  Clone and Auto-Sync
    # ------------------------------------------
    print("\n3️⃣ Cloning voices and Auto-Syncing French dialogue! (This takes a moment...)")
    total_duration_ms = int(data[-1]["end"] * 1000) + 15000 
    master_french_audio = AudioSegment.silent(duration=total_duration_ms)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

    for i, line in enumerate(data):
        spk = line["speaker"]
        french_text = line["french_translation"]
        
        actual_start_ms = int(line["start"] * 1000)
        target_duration_sec = line["end"] - line["start"]
            
        current_ref = f"{data_dir}/ref_{spk}.wav"
        raw_path = f"{data_dir}/raw_line_{i}.wav"
        
        print(f"    [{spk}]: {french_text}")
        
        estimated_natural_duration = len(french_text) / 14.0
        ai_speed = estimated_natural_duration / target_duration_sec
        ai_speed = max(1.0, min(ai_speed, 1.30)) 
        
        tts.tts_to_file(
            text=french_text, 
            speaker_wav=current_ref, 
            language="fr", 
            file_path=raw_path,
            speed=ai_speed 
        )
        
        temp_audio = AudioSegment.from_wav(raw_path)
        nonsilent_ranges = detect_nonsilent(temp_audio, min_silence_len=50, silence_thresh=-38)
        
        if nonsilent_ranges:
            trim_start = nonsilent_ranges[0][0]
            trim_end = nonsilent_ranges[-1][1]
            trimmed_audio = temp_audio[trim_start:trim_end]
        else:
            trimmed_audio = temp_audio
            
        if i + 1 < len(data):
            next_start_ms = int(data[i+1]["start"] * 1000)
            is_same_speaker = data[i+1]["speaker"] == spk
            
            if is_same_speaker:
                gap_duration_ms = (next_start_ms - actual_start_ms) + 1000
            else:
                gap_duration_ms = (next_start_ms - actual_start_ms) + 200
                
            max_safe_duration_ms = max(500, gap_duration_ms)
        else:
            max_safe_duration_ms = len(trimmed_audio) + 3000

        if len(trimmed_audio) > max_safe_duration_ms:
            speed_ratio = len(trimmed_audio) / max_safe_duration_ms
            safe_ratio = min(speed_ratio, 1.5)
            
            trimmed_audio = speedup(trimmed_audio, playback_speed=safe_ratio, chunk_size=30, crossfade=15)
            
            if len(trimmed_audio) > max_safe_duration_ms:
                trimmed_audio = trimmed_audio[:max_safe_duration_ms].fade_out(100)
            
            print(f"     Squished audio to fit (Ratio: {safe_ratio:.2f}x).")
        else:
            print(f"     Audio fits naturally.")

        trimmed_audio = trimmed_audio.fade_in(30).fade_out(30)
        master_french_audio = master_french_audio.overlay(trimmed_audio, position=actual_start_ms)

    master_french_audio = master_french_audio[:len(base_audio)]
    
    french_only_path = f"{data_dir}/final_french_dub_SYNCED.wav"
    master_french_audio.export(french_only_path, format="wav")
    print(f"\n SYNC COMPLETE! The fully synced audio is saved as {french_only_path}")

    # ------------------------------------------
    # High-Fidelity Background Extraction
    # ------------------------------------------
    print("\n Extracting high-fidelity background (Music & Effects)...")
    original_full_audio = f"{data_dir}/original_full_audio.wav"
    os.system(f"ffmpeg -i {shared_video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {original_full_audio} -y -loglevel error")
    
    subprocess.run(["demucs", "-d", "cuda", "--two-stems", "vocals", "-o", data_dir, original_full_audio], check=True)
    background_path = glob.glob(f"{data_dir}/htdemucs/original_full_audio/no_vocals.*")[0]

    # ------------------------------------------
    # Final Cinematic Mixing
    # ------------------------------------------
    print("\n Finalizing the mix: Merging French voices with Background music...")
    final_video_output = f"{data_dir}/ULTIMATE_DUBBED_VIDEO.mp4"
    
    os.system(f"""
    ffmpeg -i {shared_video_path} -i {french_only_path} -i {background_path} \
    -filter_complex "[1:a][2:a]amix=inputs=2:duration=first[aout]" \
    -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k \
    {final_video_output} -y -loglevel error
    """)

    print(f" PROJECT COMPLETE! Check {final_video_output}")

    # ==========================================
    #  THE VRAM ASSASSINATION
    # ==========================================
    print(" Wiping XTTS from GPU memory...")
    del tts
    gc.collect()
    torch.cuda.empty_cache()

    return final_video_output

# ==========================================
# THE GRADIO WEB UI
# ==========================================
with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
    gr.Markdown("#  Master Control: AI Video Dubbing Studio")
    gr.Markdown("Upload a video. This UI will orchestrate Service A (Extraction) and Service B (Translation) before running Service C (XTTS Voice Cloning) to produce the final movie.")
    
    with gr.Row():
        vid_in = gr.Video(label="Upload Original Video")
        vid_out = gr.Video(label="Final Cinematic French Dub")
        
    btn = gr.Button(" Execute Master Pipeline", variant="primary")
    btn.click(fn=process_pipeline, inputs=vid_in, outputs=vid_out)

ui.queue().launch(server_name="0.0.0.0", server_port=7860)