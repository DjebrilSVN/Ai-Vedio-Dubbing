import gradio as gr
import os
import json
import glob
import subprocess
import warnings
import torch
import shutil
import gc 
import re
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from TTS.api import TTS
from TTS.utils.manage import ModelManager 
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download, login

warnings.filterwarnings("ignore")

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

# --- Python 3.11 / 3.12 coqpit Generic Type Fix ---
# issubclass() crashes on generic types like Optional[str]. Two-layer fix:
# 1. Patch issubclass in the module namespace
# 2. Wrap _deserialize to catch TypeError at the call site (the reliable fix)
try:
    import coqpit.coqpit as _coqpit_module
    import builtins as _builtins

    _orig_issubclass = _builtins.issubclass
    def _safe_issubclass(cls, classinfo):
        try:
            return _orig_issubclass(cls, classinfo)
        except TypeError:
            return False
    _coqpit_module.issubclass = _safe_issubclass

    _orig_deserialize = _coqpit_module._deserialize
    def _safe_deserialize(x, field_type):
        try:
            return _orig_deserialize(x, field_type)
        except TypeError:
            return x
        except ValueError as e:
            if "does not match" in str(e):
                return x
            raise e
    _coqpit_module._deserialize = _safe_deserialize
except Exception:
    pass

# ==========================================
# PRE-DOWNLOAD DIRECTLY TO DISK ON BOOT
# ==========================================
print("Checking/Downloading Models to disk (This may take a while)...")
model_name = "facebook/nllb-200-distilled-1.3B"
snapshot_download(repo_id=model_name)

xtts_path = snapshot_download(repo_id="coqui/XTTS-v2")

print("Caching Demucs Audio Separation Models...")
try:
    import demucs.pretrained
    demucs.pretrained.get_model('htdemucs')
except Exception as e:
    print(f"Demucs cache failed: {e}")

print("Models securely cached on your hard drive.")

# --- UNIVERSAL LANGUAGE OMNIROUTER ---
# max/min/pause/punct  → sentence grouper tuning
# whisper/temperature/atempo/silence_floor/min_slot → Advanced Parameters (UI sliders, auto-populate on lang change)
LANGUAGE_MAP = {
    # Latin-script European: standard speech rates, moderate settings
    "Spanish":    {"nllb": "spa_Latn", "xtts": "es",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50},
    "French":     {"nllb": "fra_Latn", "xtts": "fr",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50},
    "German":     {"nllb": "deu_Latn", "xtts": "de",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.40, "silence_floor": -38, "min_slot": 100},
    "Italian":    {"nllb": "ita_Latn", "xtts": "it",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50},
    "Portuguese": {"nllb": "por_Latn", "xtts": "pt",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50},
    "Polish":     {"nllb": "pol_Latn", "xtts": "pl",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50},
    "Turkish":    {"nllb": "tur_Latn", "xtts": "tr",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50},
    "Russian":    {"nllb": "rus_Cyrl", "xtts": "ru",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50},
    "Dutch":      {"nllb": "nld_Latn", "xtts": "nl",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50},
    "Czech":      {"nllb": "ces_Latn", "xtts": "cs",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50},
    "Hungarian":  {"nllb": "hun_Latn", "xtts": "hu",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50},
    # Special scripts: longer phoneme duration, need more breathing room
    "Arabic":     {"nllb": "arb_Arab", "xtts": "ar",    "whisper": "large-v3", "temperature": 0.70, "atempo": 1.35, "silence_floor": -40, "min_slot": 150},
    "Chinese (Mandarin)": {"nllb": "zho_Hans", "xtts": "zh-cn", "whisper": "large-v3", "temperature": 0.70, "atempo": 1.25, "silence_floor": -40, "min_slot": 150},
    "Japanese":   {"nllb": "jpn_Jpan", "xtts": "ja",    "whisper": "large-v3", "temperature": 0.70, "atempo": 1.25, "silence_floor": -40, "min_slot": 150},
    "Korean":     {"nllb": "kor_Hang", "xtts": "ko",    "whisper": "large-v3", "temperature": 0.70, "atempo": 1.30, "silence_floor": -40, "min_slot": 150},
}

def run_extraction(video_filename: str, hf_user_token: str, whisper_model_choice: str = "large-v3"):
    print(f"\nPhase 1: Extraction & Diarization for {video_filename}")
    data_dir = "/app/data"
    video_path = f"{data_dir}/{video_filename}"
    raw_audio_path = f"{data_dir}/raw_audio.wav"
    demucs_out_dir = f"{data_dir}/separated"
    final_json_path = f"{data_dir}/final_master_transcript.json"

    print("Extracting raw audio...")
    os.system(f"/usr/bin/ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {raw_audio_path} -y -loglevel error")

    print("Isolating vocals to prevent AI hallucinations...")
    subprocess.run(["python", "-m", "demucs.separate", "-n", "htdemucs", "--two-stems", "vocals", "-d", "cuda", "-o", demucs_out_dir, raw_audio_path], check=True)
    clean_vocal_path = glob.glob(f"{demucs_out_dir}/htdemucs/raw_audio/vocals.*")[0]
    shutil.copy(clean_vocal_path, f"{data_dir}/clean_vocals.wav")

    print("Loading Pyannote model into memory...")
    if hf_user_token and hf_user_token.strip():
        login(hf_user_token.strip())
    else:
        raise ValueError("No Hugging Face token provided.") 
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(torch.device("cuda"))
    pipeline.instantiate({"clustering": {"threshold": 0.60}})
    
    print("Listening to PURIFIED audio with tuned sensitivity...")
    raw_diarization = pipeline(clean_vocal_path, min_speakers=1, max_speakers=5)

    diarization = Annotation()
    for turn, _, speaker in raw_diarization.itertracks(yield_label=True):
        if turn.end - turn.start > 0.2:
            diarization[turn] = speaker

    print("Building dynamic voice references...")
    clean_vocals = AudioSegment.from_file(clean_vocal_path)
    unique_speakers = list(set([speaker for _, _, speaker in raw_diarization.itertracks(yield_label=True)]))
    for target_speaker in unique_speakers:
        longest_duration = 0
        best_start, best_end = 0, 0
        for turn, _, speaker in raw_diarization.itertracks(yield_label=True):
            if speaker == target_speaker:
                duration = turn.end - turn.start
                if duration > longest_duration:
                    longest_duration = duration
                    best_start, best_end = turn.start, turn.end
        start_ms, end_ms = int(best_start * 1000), int(best_end * 1000)
        if (end_ms - start_ms) > 6000: end_ms = start_ms + 6000
        ref_filename = f"{data_dir}/ref_{target_speaker}.wav"
        clean_vocals[start_ms:end_ms].export(ref_filename, format="wav")

    print("Wiping Pyannote from GPU memory...")
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Loading Whisper ({whisper_model_choice}) to transcribe audio...")
    whisper_model = WhisperModel(whisper_model_choice, device="cuda", compute_type="float16")
    segments, info = whisper_model.transcribe(
        video_path, 
        word_timestamps=True, 
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    print("Merging words with your speaker labels...")
    final_pipeline_data = []
    last_known_speaker = "SPEAKER_00"

    for segment in segments:
        for word in segment.words:
            word_start = round(word.start, 2)
            word_end = round(word.end, 2)
            word_midpoint = (word_start + word_end) / 2.0
            
            current_speaker = "UNKNOWN"
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if (turn.start - 0.2) <= word_midpoint <= (turn.end + 0.2):
                    current_speaker = speaker
                    last_known_speaker = speaker
                    break 
                    
            if current_speaker == "UNKNOWN":
                current_speaker = last_known_speaker
                
            word_data = {
                "word": word.word.strip(),
                "start": word_start,
                "end": word_end,
                "speaker": current_speaker
            }
            final_pipeline_data.append(word_data)
            print(f"[{word_start}s -> {word_end}s] {current_speaker}: {word_data['word']}")

    print("Wiping Whisper from GPU memory...")
    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_pipeline_data, f, indent=4, ensure_ascii=False)
    
    return final_json_path

def run_translation(transcript_filepath: str, target_lang_name: str):
    print("\nPhase 2: Translation for transcripts...")
    data_dir = "/app/data"
    output_filename = f"{data_dir}/translated_transcript_fixed.json"

    nllb_code = LANGUAGE_MAP[target_lang_name]["nllb"]

    print(f"Loading NLLB-200-1.3B into GPU memory targeting {target_lang_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main", local_files_only=False).to("cuda")
    target_id = tokenizer.convert_tokens_to_ids(nllb_code)

    with open(transcript_filepath, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    for entry in transcript_data:
        if "speaker" not in entry:
            entry["speaker"] = "UNKNOWN_SPEAKER"

    MAX_PAUSE_SECONDS = 0.4
    MAX_CHARS = 150   # Last-resort fallback only — primary breaks are punctuation, pause, speaker change
    MIN_CHARS = 5

    grouped_sentences = []
    current_sentence = {
        "speaker": transcript_data[0].get("speaker", "SPEAKER_00"), 
        "text": transcript_data[0].get("word", ""), 
        "start": transcript_data[0].get("start", 0.0), 
        "end": transcript_data[0].get("end", 0.0)
    }

    for word_data in transcript_data[1:]:
        time_gap = word_data["start"] - current_sentence["end"]
        text = current_sentence["text"].strip()
        is_end_of_sentence = text.endswith(('.', '?', '!'))
        if re.search(r'\b(Mr|Ms|Mrs|Dr|M)\.?$', text, re.IGNORECASE):
            is_end_of_sentence = False
        speaker_changed = word_data["speaker"] != current_sentence["speaker"]
        
        must_cut = speaker_changed or (is_end_of_sentence and len(text) > MIN_CHARS) or time_gap > MAX_PAUSE_SECONDS or len(text) > MAX_CHARS
        if not must_cut:
            current_sentence["text"] += " " + word_data["word"]
            current_sentence["end"] = word_data["end"] 
        else:
            grouped_sentences.append(current_sentence)
            current_sentence = {"speaker": word_data["speaker"], "text": word_data["word"], "start": word_data["start"], "end": word_data["end"]}

    grouped_sentences.append(current_sentence)

    print("Processing Deep Translation via Beam Search...")
    translated_data = []
    for block in grouped_sentences:
        original_text = block["text"].strip()
        clean_text = original_text.replace("♪", "").strip()
        if not clean_text: continue
            
        inputs = tokenizer(clean_text, return_tensors="pt").to("cuda")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=target_id, max_length=100, num_beams=5, repetition_penalty=1.2)
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        print(f"[{block['start']}s] {block['speaker']}: {translated_text}")
        translated_data.append({"speaker": block["speaker"], "start": block["start"], "end": block["end"], "english_original": original_text, "translated_text": translated_text})

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=4, ensure_ascii=False)

    print("Wiping NLLB from GPU memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_filename

def format_timestamp(seconds: float) -> str:
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    m = s // 60
    h = m // 60
    return f"{h:02d}:{m%60:02d}:{s%60:02d},{ms:03d}"

def process_pipeline(youtube_url, uploaded_video_path, target_lang_name, burn_subtitles, hf_user_token,
                     whisper_model_choice="large-v3", xtts_temperature=0.65, max_atempo=1.5,
                     silence_floor=-38, min_slot_ms=50):
    yield "Initializing Pipeline...", None

    if not hf_user_token or hf_user_token.strip() == "":
        yield "❌ SECURITY HALT: You must provide your Hugging Face Access Token!", None
        return
    
    data_dir = "/app/data"
    
    yield "Purging old video data (Safeguarding models)...", None
    import os, shutil
    for item in os.listdir(data_dir):
        if item == "models_cache":
            continue
        item_path = os.path.join(data_dir, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path, ignore_errors=True)
            else:
                os.remove(item_path)
        except Exception:
            pass
            
    shared_video_path = f"{data_dir}/source_video.mp4"
    
    if youtube_url and youtube_url.strip() != "":
        yield "Fetching video straight from YouTube...", None
        if os.path.exists(shared_video_path): os.remove(shared_video_path)
        os.system(f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' {youtube_url} -o {shared_video_path} -N 4")
        if not os.path.exists(shared_video_path):
            yield "FAILED: Failed to download from YouTube.", None
            return
    elif uploaded_video_path is not None:
        yield "Saving uploaded video to shared volume...", None
        shutil.copy(uploaded_video_path, shared_video_path)
    else:
        yield "FAILED: No video provided! Ignoring request.", None
        return
        
    yield "Phase 1: High-Fidelity Audio Extraction...", None
    transcript_path = run_extraction("source_video.mp4", hf_user_token, whisper_model_choice)
    
    yield f"Phase 2: Neural Translation to {target_lang_name}...", None
    translated_json_path = run_translation(transcript_path, target_lang_name)
    
    yield "Phase 3: Auto-Syncing & Cloning Voices...", None
    
    with open(translated_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    subs_path = f"{data_dir}/subs.srt"
    if burn_subtitles:
        with open(subs_path, "w", encoding="utf-8") as srt_file:
            for idx, line in enumerate(data):
                start_str = format_timestamp(line["start"])
                end_str = format_timestamp(line["end"])
                srt_file.write(f"{idx+1}\n{start_str} --> {end_str}\n{line['translated_text']}\n\n")

    base_audio_path = f"{data_dir}/base_audio.wav"
    os.system(f"/usr/bin/ffmpeg -i {shared_video_path} -vn -acodec pcm_s16le -ar 22050 -ac 1 {base_audio_path} -y -loglevel error")
    base_audio = AudioSegment.from_wav(base_audio_path)

    subprocess.run(["demucs", "-d", "cuda", "--two-stems", "vocals", "-o", data_dir, base_audio_path], check=True)
    vocal_stem_path = f"{data_dir}/htdemucs/base_audio/vocals.wav"
    if not os.path.exists(vocal_stem_path):
        vocal_stem_path = glob.glob(f"{data_dir}/*/base_audio/vocals.wav")[0]
    clean_vocals = AudioSegment.from_wav(vocal_stem_path)

    refs = {}
    for line in data:
        spk = line["speaker"]
        duration = line["end"] - line["start"]
        if spk not in refs: refs[spk] = {"start": line["start"], "end": line["end"], "duration": duration}
        else:
            if abs(duration - 6.0) < abs(refs[spk]["duration"] - 6.0):
                refs[spk] = {"start": line["start"], "end": line["end"], "duration": duration}

    for spk, info in refs.items():
        start_ms, end_ms = int(info["start"] * 1000), int(info["end"] * 1000)
        ref_path = f"{data_dir}/ref_{spk}.wav"
        clean_vocals[start_ms:end_ms].export(ref_path, format="wav")

    total_duration_ms = int(data[-1]["end"] * 1000) + 15000 
    master_dub_audio = AudioSegment.silent(duration=total_duration_ms)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    xtts_lang_code = LANGUAGE_MAP[target_lang_name]["xtts"]

    for i, line in enumerate(data):
        yield f"Generating dialogue block {i+1}/{len(data)}...", None
        spk = line["speaker"]
        translated_text = line["translated_text"]
        actual_start_ms = int(line["start"] * 1000)
        
        current_ref = f"{data_dir}/ref_{spk}.wav"
        raw_path = f"{data_dir}/raw_line_{i}.wav"
        
        ai_speed = 1.0 
        
        tts.tts_to_file(
            text=translated_text, 
            speaker_wav=current_ref, 
            language=xtts_lang_code, 
            file_path=raw_path,
            speed=ai_speed,
            temperature=xtts_temperature
        )
        temp_audio = AudioSegment.from_wav(raw_path)
        nonsilent_ranges = detect_nonsilent(temp_audio, min_silence_len=50, silence_thresh=silence_floor)
        trimmed_audio = temp_audio[nonsilent_ranges[0][0]:nonsilent_ranges[-1][1]] if nonsilent_ranges else temp_audio
            
        if i + 1 < len(data):
            next_start_ms = int(data[i+1]["start"] * 1000)
            gap_duration_ms = (next_start_ms - actual_start_ms)
            max_safe_duration_ms = max(int(min_slot_ms), gap_duration_ms)
        else:
            max_safe_duration_ms = len(trimmed_audio) + 3000

        if len(trimmed_audio) > max_safe_duration_ms:
            speed_ratio = len(trimmed_audio) / max_safe_duration_ms
            safe_ratio = min(speed_ratio, max_atempo)
            
            temp_squish_in = f"{data_dir}/temp_squish_in.wav"
            temp_squish_out = f"{data_dir}/temp_squish_out.wav"
            trimmed_audio.export(temp_squish_in, format="wav")
            os.system(f"/usr/bin/ffmpeg -i {temp_squish_in} -filter:a atempo={safe_ratio} {temp_squish_out} -y -loglevel error")
            trimmed_audio = AudioSegment.from_wav(temp_squish_out)
            
            if len(trimmed_audio) > max_safe_duration_ms:
                trimmed_audio = trimmed_audio[:max_safe_duration_ms].fade_out(100)
        
        trimmed_audio = trimmed_audio.fade_in(30).fade_out(30)
        master_dub_audio = master_dub_audio.overlay(trimmed_audio, position=actual_start_ms)

    master_dub_audio = master_dub_audio[:len(base_audio)]
    dub_only_path = f"{data_dir}/final_dub_SYNCED.wav"
    master_dub_audio.export(dub_only_path, format="wav")

    yield "Re-integrating Original Background Audio & FX...", None
    original_full_audio = f"{data_dir}/original_full_audio.wav"
    os.system(f"/usr/bin/ffmpeg -i {shared_video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {original_full_audio} -y -loglevel error")
    subprocess.run(["demucs", "-d", "cuda", "--two-stems", "vocals", "-o", data_dir, original_full_audio], check=True)
    background_path = glob.glob(f"{data_dir}/htdemucs/original_full_audio/no_vocals.*")[0]

    yield "Performing Final Video Mix...", None
    final_video_output = f"{data_dir}/ULTIMATE_DUBBED_VIDEO.mp4"
    
    if burn_subtitles:
        subtitle_filter = f"-vf \"subtitles={subs_path}\""
        video_codec = "-c:v libx264"
    else:
        subtitle_filter = ""
        video_codec = "-c:v copy"
    
    os.system(f"""
    /usr/bin/ffmpeg -i {shared_video_path} -i {dub_only_path} -i {background_path} \
    -filter_complex "[1:a][2:a]amix=inputs=2:duration=first[aout]" \
    -map 0:v -map "[aout]" {subtitle_filter} {video_codec} -c:a aac -b:a 192k \
    {final_video_output} -y -loglevel error
    """)

    del tts
    gc.collect()
    torch.cuda.empty_cache()

    yield "PROJECT COMPLETE! Video is Ready.", final_video_output

# ==========================================
# MODERN NATIVE UI - LEGENDARY THEME
# ==========================================

custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
).set(
    body_background_fill="*neutral_950",
    body_text_color="*neutral_100",
    background_fill_primary="*neutral_900",
    background_fill_secondary="*neutral_800",
    border_color_primary="*neutral_700",
    block_background_fill="*neutral_800",
    block_label_background_fill="*primary_600",
    block_label_text_color="*neutral_100",
    block_title_text_color="*neutral_100",
    block_radius="*radius_xl",
    block_border_width="1px",
    input_background_fill="*neutral_700",
    button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_400, *secondary_400)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
)

css = """
.legendary-header {
    text-align: center;
    background: linear-gradient(90deg, #a855f7, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.legendary-sub {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}
/* Style strictly the custom status log without touching other inputs */
#status-log textarea {
    font-family: 'Consolas', 'Courier New', monospace !important;
    color: #4ade80 !important;
    background-color: #0f1016 !important;
    border: 1px solid #1e202e !important;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.5) !important;
}
"""

with gr.Blocks(theme=custom_theme, css=css) as ui:
    gr.HTML("<h1 class='legendary-header'>AI Video Dubbing Studio</h1>")
    gr.HTML("<div class='legendary-sub'>Zero-bloat GPU container pipeline with high-fidelity voice cloning, translation routing, and automated syncing.</div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Input Source")
                hf_token = gr.Textbox(
                    label="🔑 Hugging Face Access Token (Required)",
                    type="password",
                    placeholder="hf_...",
                    info="Required for speaker diarization. Get yours at huggingface.co/settings/tokens"
                )
                yt_url = gr.Textbox(label="YouTube Link (Optional)", placeholder="Paste a URL here to auto-download...")
                vid_in = gr.Video(label="OR Upload Original Video")
            
            with gr.Group():
                gr.Markdown("### Output Settings")
                target_lang = gr.Dropdown(
                    choices=list(LANGUAGE_MAP.keys()), 
                    value="French", 
                    label="Target Language",
                    interactive=True
                )
                burn_subs = gr.Checkbox(label="Burn Translated Subtitles onto Video", value=True)

            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                gr.Markdown("*All defaults match the tuned production settings. Only change if you know what you're doing.*")
                whisper_choice = gr.Radio(
                    choices=["tiny", "base", "medium", "large-v2", "large-v3"],
                    value="large-v3",
                    label="Whisper Model",
                    info="large-v3 = best quality (default). base/medium = faster for long videos."
                )
                xtts_temp = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.65, step=0.05,
                    label="XTTS Temperature",
                    info="0.65 = default. Lower → more robotic but stable. Higher → more expressive but may hallucinate."
                )
                atempo_max = gr.Slider(
                    minimum=1.0, maximum=2.0, value=1.5, step=0.05,
                    label="Max Speed-Up (atempo)",
                    info="1.5 = default. Max compression ratio before hard-cutting overflow audio."
                )
                sil_floor = gr.Slider(
                    minimum=-60, maximum=-20, value=-38, step=1,
                    label="Silence Detection Floor (dBFS)",
                    info="-38 = default. Lower = trim more silence. Higher = keep more."
                )
                min_slot = gr.Slider(
                    minimum=0, maximum=1000, value=50, step=50,
                    label="Minimum Slot Size (ms)",
                    info="50ms = default. Hard floor for audio slot budget."
                )
                
            btn = gr.Button("Execute Cinematic Dubbing", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### Processing Interface")
            status_out = gr.Textbox(label="Live Progression Pipeline", lines=12, interactive=False, elem_id="status-log")
            vid_out = gr.Video(label="Final Rendered Output")
        
    btn.click(
        fn=process_pipeline, 
        inputs=[yt_url, vid_in, target_lang, burn_subs, hf_token,
                whisper_choice, xtts_temp, atempo_max, sil_floor, min_slot], 
        outputs=[status_out, vid_out]
    )

    # When language changes, auto-update Advanced Parameter sliders to that language's defaults
    def update_lang_defaults(lang_name):
        d = LANGUAGE_MAP[lang_name]
        return d["whisper"], d["temperature"], d["atempo"], d["silence_floor"], d["min_slot"]

    target_lang.change(
        fn=update_lang_defaults,
        inputs=[target_lang],
        outputs=[whisper_choice, xtts_temp, atempo_max, sil_floor, min_slot]
    )

ui.queue().launch(server_name="0.0.0.0", server_port=7860)
