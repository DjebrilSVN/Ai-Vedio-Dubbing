import gradio as gr
import os
os.environ["COQUI_TOS_AGREED"] = "1"
import json
import glob
import subprocess
import warnings
import torch
import shutil
import gc 
import re
import sys

# --- Python 3.12 `imp` Module Shim for Legacy Libraries ---
if "imp" not in sys.modules:
    import importlib.util
    class MockImp:
        def reload(self, module):
            import importlib
            return importlib.reload(module)
        def load_source(self, name, pathname):
            spec = importlib.util.spec_from_file_location(name, pathname)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    sys.modules["imp"] = MockImp()

# --- Numpy 2.x Backward Compatibility Shim ---
try:
    import numpy as np
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'complex'): np.complex = complex
    if not hasattr(np, 'object'): np.object = object
except ImportError:
    pass

# --- Torchaudio 2.4+ Backend Shim for Pyannote ---
import torchaudio
if not hasattr(torchaudio, 'AudioMetaData'):
    torchaudio.AudioMetaData = type('AudioMetaData', (), {})
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox_io"]
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

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
os.makedirs("./data", exist_ok=True)

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

LANGUAGE_MAP = {
    "English":    {"nllb": "eng_Latn", "xtts": "en",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Spanish":    {"nllb": "spa_Latn", "xtts": "es",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "French":     {"nllb": "fra_Latn", "xtts": "fr",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "German":     {"nllb": "deu_Latn", "xtts": "de",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.40, "silence_floor": -38, "min_slot": 100, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Italian":    {"nllb": "ita_Latn", "xtts": "it",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Portuguese": {"nllb": "por_Latn", "xtts": "pt",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Polish":     {"nllb": "pol_Latn", "xtts": "pl",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Turkish":    {"nllb": "tur_Latn", "xtts": "tr",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Russian":    {"nllb": "rus_Cyrl", "xtts": "ru",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.50, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Dutch":      {"nllb": "nld_Latn", "xtts": "nl",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Czech":      {"nllb": "ces_Latn", "xtts": "cs",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Hungarian":  {"nllb": "hun_Latn", "xtts": "hu",    "whisper": "large-v3", "temperature": 0.65, "atempo": 1.45, "silence_floor": -38, "min_slot": 50, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Arabic":     {"nllb": "arb_Arab", "xtts": "ar",    "whisper": "large-v3", "temperature": 0.70, "atempo": 1.35, "silence_floor": -40, "min_slot": 150, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Chinese (Mandarin)": {"nllb": "zho_Hans", "xtts": "zh-cn", "whisper": "large-v3", "temperature": 0.70, "atempo": 1.25, "silence_floor": -40, "min_slot": 150, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Japanese":   {"nllb": "jpn_Jpan", "xtts": "ja",    "whisper": "large-v3", "temperature": 0.70, "atempo": 1.25, "silence_floor": -40, "min_slot": 150, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
    "Korean":     {"nllb": "kor_Hang", "xtts": "ko",    "whisper": "large-v3", "temperature": 0.70, "atempo": 1.30, "silence_floor": -40, "min_slot": 150, "normalize": True, "vol_target": -16.0, "nr_level": 15, "hp_freq": 80, "lp_freq": 10000},
}

def run_extraction(video_filename: str, hf_user_token: str, whisper_model_choice: str = "large-v3"):
    print(f"\nPhase 1: Extraction & Diarization for {video_filename}")
    data_dir = "./data"
    video_path = f"{data_dir}/{video_filename}"
    raw_audio_path = f"{data_dir}/raw_audio.wav"
    demucs_out_dir = f"{data_dir}/separated"
    final_json_path = f"{data_dir}/final_master_transcript.json"

    print("Extracting raw audio...")
    os.system(f"/usr/bin/ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {raw_audio_path} -y -loglevel error")

    print("Isolating vocals...")
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
    
    print("Listening to PURIFIED audio...")
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

    print(f"Loading Whisper ({whisper_model_choice})...")
    whisper_model = WhisperModel(whisper_model_choice, device="cuda", compute_type="float16")
    segments, info = whisper_model.transcribe(
        video_path, 
        word_timestamps=True, 
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    print("Merging words with speaker labels...")
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

    print("Wiping Whisper from GPU memory...")
    del whisper_model
    gc.collect()
    torch.cuda.empty_cache()

    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_pipeline_data, f, indent=4, ensure_ascii=False)
    
    return final_json_path

def run_translation(transcript_filepath: str, target_lang_name: str):
    print("\nPhase 2: Translation for transcripts...")
    data_dir = "./data"
    output_filename = f"{data_dir}/translated_transcript_fixed.json"

    nllb_code = LANGUAGE_MAP[target_lang_name]["nllb"]

    print(f"Loading NLLB-200-1.3B targeting {target_lang_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision="main", local_files_only=False).to("cuda")
    target_id = tokenizer.convert_tokens_to_ids(nllb_code)

    with open(transcript_filepath, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    for entry in transcript_data:
        if "speaker" not in entry:
            entry["speaker"] = "UNKNOWN_SPEAKER"

    MAX_PAUSE_SECONDS = 0.4  
    MAX_CHARS = 150 
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

    print("Processing Deep Translation...")
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
                     silence_floor=-38, min_slot_ms=50,
                     normalize_audio=True, target_volume=-16.0, noise_reduction=15, high_pass=80, low_pass=10000):
    yield "Initializing Pipeline...", None

    if not hf_user_token or str(hf_user_token).strip() == "":
        yield "❌ SECURITY HALT: You must provide your Hugging Face Access Token!", None
        return
    
    data_dir = "./data"
    
    yield "Purging old video data (Safeguarding models)...", None
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
    
    # Bug fix: Handle dynamic logic correctly for youtube url or fallback to upload
    # Ensure any valid truthy is stripped safely
    is_yt = isinstance(youtube_url, str) and youtube_url.strip() != ""
    if is_yt:
        yield "Fetching video straight from YouTube...", None
        if os.path.exists(shared_video_path): os.remove(shared_video_path)
        safe_url = youtube_url.strip().replace('"', '\\"') # basic safety
        os.system(f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' \"{safe_url}\" -o {shared_video_path} -N 4")
        if not os.path.exists(shared_video_path):
            yield "FAILED: Failed to download from YouTube. Check URL.", None
            return
    elif uploaded_video_path is not None and str(uploaded_video_path).strip() != "":
        yield "Saving uploaded video to shared volume...", None
        try:
            shutil.copy(str(uploaded_video_path), shared_video_path)
        except Exception as e:
            yield f"FAILED: Copy uploaded video error: {e}", None
            return
    else:
        yield "FAILED: No video provided! Ignoring request.", None
        return
        
    yield "Phase 1: High-Fidelity Audio Extraction...", None
    try:
        transcript_path = run_extraction("source_video.mp4", hf_user_token, whisper_model_choice)
    except Exception as e:
        yield f"FAILED: Processing error during extraction: {e}", None
        return
    
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
        
        filter_chain = []
        if high_pass > 0:
            filter_chain.append(f"highpass=f={high_pass}")
        if low_pass < 15000:
            filter_chain.append(f"lowpass=f={low_pass}")
        if noise_reduction > 0:
            filter_chain.append(f"afftdn=nr={noise_reduction}")
        
        if filter_chain:
            filtered_path = f"{data_dir}/filtered_line_{i}.wav"
            af_val = ",".join(filter_chain)
            os.system(f"/usr/bin/ffmpeg -i {raw_path} -af \"{af_val}\" {filtered_path} -y -loglevel error")
            temp_audio = AudioSegment.from_wav(filtered_path)
        else:
            temp_audio = AudioSegment.from_wav(raw_path)
            
        nonsilent_ranges = detect_nonsilent(temp_audio, min_silence_len=50, silence_thresh=silence_floor)
        trimmed_audio = temp_audio[nonsilent_ranges[0][0]:nonsilent_ranges[-1][1]] if nonsilent_ranges else temp_audio

        if normalize_audio and trimmed_audio.dBFS != float('-inf'):
            change_in_dBFS = target_volume - trimmed_audio.dBFS
            if change_in_dBFS < 20.0:
                trimmed_audio = trimmed_audio.apply_gain(change_in_dBFS)
            
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
        video_codec = "-c:v libx264 -preset ultrafast"
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
# MODERN PORTFOLIO VIBE UI
# ==========================================

custom_theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#0A0910",
    body_background_fill_dark="#0A0910",
    body_text_color="#FFFFFF",
    body_text_color_dark="#FFFFFF",
    background_fill_primary="#12111A",
    background_fill_primary_dark="#12111A",
    background_fill_secondary="#0A0910",
    background_fill_secondary_dark="#0A0910",
    border_color_primary="#1E1D2B",
    border_color_primary_dark="#1E1D2B",
    block_background_fill="#12111A",
    block_background_fill_dark="#12111A",
    block_border_width="1px",
    block_label_background_fill="#1A1826",
    block_label_background_fill_dark="#1A1826",
    block_label_text_color="#94A3B8",
    block_label_text_color_dark="#94A3B8",
    block_title_text_color="#FFFFFF",
    block_title_text_color_dark="#FFFFFF",
    input_background_fill="#1A1826",
    input_background_fill_dark="#1A1826",
    button_primary_background_fill="linear-gradient(90deg, #A855F7, #D946EF)",
    button_primary_background_fill_dark="linear-gradient(90deg, #A855F7, #D946EF)",
    button_primary_background_fill_hover="linear-gradient(90deg, #9333EA, #C026D3)",
    button_primary_background_fill_hover_dark="linear-gradient(90deg, #9333EA, #C026D3)",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#1A1826",
    button_secondary_background_fill_dark="#1A1826",
    button_secondary_text_color="#FFFFFF",
    button_secondary_text_color_dark="#FFFFFF",
    slider_color="#A855F7",
    slider_color_dark="#A855F7"
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── KEYFRAMES ──────────────────────────────── */
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes slideDown {
  from { opacity: 0; transform: translateY(-10px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%      { opacity: 0.5; }
}

/* ── GLOBAL ─────────────────────────────────── */
html, body, gradio-app, .gradio-container, #root, .wrap {
    background: transparent !important;
    background-color: transparent !important;
}
html {
    background-color: #08070D !important;
    min-height: 100vh;
}
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
footer { display: none !important; }

/* ── FLUID ANIMATED BACKGROUNDS ──────────────── */
body::before {
    content: ''; position: fixed; top: -10%; left: -20%; width: 70vw; height: 70vh; border-radius: 50%;
    background: radial-gradient(circle, rgba(124, 58, 237, 0.25) 0%, rgba(124, 58, 237, 0) 65%);
    filter: blur(120px); z-index: -100; animation: floatOrb 15s infinite alternate ease-in-out; pointer-events: none;
}
body::after {
    content: ''; position: fixed; bottom: -20%; right: -10%; width: 80vw; height: 80vh; border-radius: 50%;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.2) 0%, rgba(16, 185, 129, 0) 65%);
    filter: blur(130px); z-index: -100; animation: floatOrb2 18s infinite alternate ease-in-out; pointer-events: none;
}
html::before {
    content: ''; position: fixed; top: 20%; left: 30%; width: 60vw; height: 60vh; border-radius: 50%;
    background: radial-gradient(circle, rgba(236, 72, 153, 0.15) 0%, rgba(236, 72, 153, 0) 65%);
    filter: blur(140px); z-index: -101; animation: floatOrb3 22s infinite alternate ease-in-out; pointer-events: none;
}

@keyframes floatOrb {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(25vw, 15vh) scale(1.3); }
}
@keyframes floatOrb2 {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(-20vw, -15vh) scale(1.2); }
}
@keyframes floatOrb3 {
    0% { transform: translate(0, 0) scale(1.1); }
    100% { transform: translate(-10vw, 25vh) scale(0.9); }
}

/* ── HERO SECTION ───────────────────────────── */
.hero-section {
    text-align: center !important;
    padding: 56px 24px 40px;
    animation: fadeIn 0.6s ease-out;
    width: 100%;
    margin: 0 auto;
    display: block;
}
.hero-section * {
    text-align: center !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(139, 92, 246, 0.1);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #A78BFA;
    margin-bottom: 24px;
}
.hero-chip-dot {
    width: 6px; height: 6px;
    background: #A78BFA;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
.hero-h1 {
    font-size: clamp(2.5rem, 5vw, 3.8rem);
    font-weight: 900;
    color: #F8FAFC;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 16px 0;
}
.hero-h1 span {
    background: linear-gradient(90deg, #C084FC, #818CF8, #C084FC);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
}
.hero-desc {
    font-size: 1.05rem;
    color: #64748B;
    line-height: 1.7;
    max-width: 520px;
    margin: 0 auto !important;
    text-align: center !important;
    display: block;
}

/* ── MAIN LAYOUT ────────────────────────────── */
.app-grid {
    max-width: 1260px;
    margin: 0 auto;
    padding: 0 28px 64px;
}

/* ── CARDS ───────────────────────────────────── */
.card {
    background: #111019 !important;
    border: 1px solid #1E1B2E !important;
    border-radius: 16px !important;
    padding: 24px 28px !important;
    margin-bottom: 16px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    animation: fadeIn 0.5s ease-out both !important;
}
.card:hover {
    border-color: #2D2747 !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4) !important;
}
.delay-1 { animation-delay: 0.08s !important; }
.delay-2 { animation-delay: 0.16s !important; }
.delay-3 { animation-delay: 0.24s !important; }
.delay-4 { animation-delay: 0.12s !important; }
.delay-5 { animation-delay: 0.20s !important; }

.card-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7C3AED;
    margin: 0 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(124, 58, 237, 0.15);
    display: flex;
    align-items: center;
    gap: 10px;
}
.card-icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 6px;
    background: rgba(124, 58, 237, 0.15);
    color: #A78BFA;
    border: 1px solid rgba(124, 58, 237, 0.3);
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.2);
}
.console-icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px; height: 26px;
    border-radius: 6px;
    background: rgba(52, 211, 153, 0.1);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.25);
    box-shadow: 0 0 10px rgba(52, 211, 153, 0.15);
}

/* ── INPUTS ──────────────────────────────────── */
.gr-input, .gr-text-input, .gr-box,
input[type=text], input[type=password], textarea, .gr-dropdown {
    background: #0B0A12 !important;
    border: 1px solid #1E1B2E !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.gr-dropdown {
    padding: 0 !important;
}
input[type=text]:focus, input[type=password]:focus, textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12) !important;
    outline: none !important;
}
input[type=text]:hover, input[type=password]:hover, textarea:hover {
    border-color: #2D2747 !important;
}
select {
    background: #0B0A12 !important;
    border: 1px solid #1E1B2E !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

label > span, .gr-label {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}
input[type=checkbox] {
    accent-color: #7C3AED !important;
}

/* ── OR DIVIDER ──────────────────────────────── */
.or-line {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 16px 0;
    font-size: 0.72rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.or-line::before, .or-line::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E1B2E;
}

/* ── UPLOAD BUTTON ───────────────────────────── */
.upload-btn {
    width: 100% !important;
    height: 48px !important;
    background: transparent !important;
    border: 1.5px dashed #2D2747 !important;
    border-radius: 12px !important;
    color: #475569 !important;
    font-size: 0.88rem !important;
    transition: all 0.25s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}
.upload-btn:hover {
    border-color: #7C3AED !important;
    color: #A78BFA !important;
    background: rgba(124, 58, 237, 0.05) !important;
}
.upload-btn::before {
    content: '';
    display: inline-block;
    width: 18px; height: 18px;
    background-color: currentColor;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") no-repeat center / contain;
}

/* ── RUN BUTTON ──────────────────────────────── */
button.primary {
    width: 100% !important;
    height: 52px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 10px !important;
}
button.primary::before {
    content: '';
    display: inline-block;
    width: 18px; height: 18px;
    background-color: currentColor;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='currentColor'%3E%3Cpolygon points='5 3 19 12 5 21 5 3'/%3E%3C/svg%3E") no-repeat center / contain;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='currentColor'%3E%3Cpolygon points='5 3 19 12 5 21 5 3'/%3E%3C/svg%3E") no-repeat center / contain;
}
button.primary {
    background: linear-gradient(135deg, #7C3AED 0%, #A855F7 50%, #C084FC 100%) !important;
    background-size: 200% auto !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.03em !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3) !important;
    margin-top: 4px !important;
}
button.primary:hover {
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.45) !important;
}
button.primary:active {
    transform: translateY(0px) !important;
}

/* ── TERMINAL/CONSOLE ────────────────────────── */
.console-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #34D399;
    margin-bottom: 12px;
}

#terminal-log textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.65 !important;
    color: #34D399 !important;
    background: #050410 !important;
    border: 1px solid rgba(52,211,153,0.1) !important;
    border-radius: 10px !important;
    padding: 16px !important;
    transition: border-color 0.3s !important;
}
#terminal-log textarea:hover {
    border-color: rgba(52,211,153,0.25) !important;
}

/* ── VIDEO OUTPUT ────────────────────────────── */
.render-box video, .render-box .video-container {
    border-radius: 12px !important;
    border: 1px solid #1E1B2E !important;
}
.render-box:hover video {
    border-color: #2D2747 !important;
}

/* ── ACCORDION ───────────────────────────────── */
.gr-accordion {
    border: none !important;
    background: transparent !important;
    margin-bottom: 12px !important;
}
.gr-accordion > .label-wrap {
    background: rgba(124, 58, 237, 0.06) !important;
    border: 1px solid rgba(124, 58, 237, 0.12) !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    color: #A78BFA !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.gr-accordion > .label-wrap:hover {
    background: rgba(124, 58, 237, 0.1) !important;
}

/* ── SLIDERS & RADIOS ────────────────────────── */
input[type=range] { accent-color: #7C3AED !important; }

/* ── MISC BLOCKS ─────────────────────────────── */
.gr-group, .gr-block, .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
"""

with gr.Blocks(theme=custom_theme, css=css) as ui:

    # ═══ HERO ═══════════════════════════════════
    gr.HTML("""
    <div class="hero-section">
        <div class="hero-chip">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #A78BFA; animation: pulse 2s ease-in-out infinite;"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/><path d="M5 3v4"/><path d="M7 5H3"/><path d="M21 17v4"/><path d="M23 19h-4"/></svg>
            AI Video Dubbing Studio
        </div>
        <h1 class="hero-h1">Dub Any Video.<br><span>In Any Language.</span></h1>
        <p class="hero-desc">Voice cloning, neural translation & automated lip-sync — production-grade, fully automated.</p>
    </div>
    """)

    # ═══ DASHBOARD ══════════════════════════════
    with gr.Row(elem_classes="app-grid"):

        # ── LEFT: Controls ──────────────────────
        with gr.Column(scale=4, min_width=400):

            # Source Material
            with gr.Column(elem_classes="card delay-1"):
                gr.HTML("""<div class="card-title"><span class="card-icon-wrap"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2" ry="2"/></svg></span> Source Material</div>""")
                yt_url = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...", lines=1)
                
                gr.HTML("<div class='or-line'>or</div>")
                upload_btn = gr.UploadButton("Upload Local Video", file_types=["video"], elem_classes="upload-btn")
                
                yt_preview = gr.HTML(visible=False)
                vid_in = gr.Video(label="Preview Container", visible=False, interactive=False)

            # Configuration
            with gr.Column(elem_classes="card delay-2"):
                gr.HTML("""<div class="card-title"><span class="card-icon-wrap"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg></span> Configuration</div>""")
                hf_token = gr.Textbox(label="Hugging Face Token", type="password", placeholder="hf_••••••••••••", lines=1)
                with gr.Row():
                    target_lang = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), value="French", label="Target Language", interactive=True, filterable=False)
                    burn_subs = gr.Checkbox(label="Burn Subtitles", value=True)

            # Advanced
            with gr.Accordion("⚙  Advanced Engine Settings", open=False):
                whisper_choice = gr.Radio(["tiny", "base", "medium", "large-v2", "large-v3"], value="large-v3", label="Whisper Model")
                with gr.Row():
                    xtts_temp  = gr.Slider(0.1, 1.0, value=0.65, step=0.05, label="Voice Temperature")
                    atempo_max = gr.Slider(1.0, 3.0, value=1.5,  step=0.05, label="Max Speed-Up")
                with gr.Row():
                    sil_floor = gr.Slider(-60, -10, value=-38, step=1,  label="Silence Floor (dBFS)")
                    min_slot  = gr.Slider(0, 1000,  value=50,  step=50, label="Min Gap (ms)")
                with gr.Row():
                    normalize_audio = gr.Checkbox(label="Normalize Audio (RMS)", value=True)
                    target_volume = gr.Slider(-35.0, -5.0, value=-16.0, step=1.0, label="Target Vol (dBFS)")
                    noise_reduction = gr.Slider(0, 50, value=15, step=1, label="Noise Reduction (dB)")
                with gr.Row():
                    high_pass = gr.Slider(0, 500, value=80, step=10, label="High-pass Filter (Hz)")
                    low_pass = gr.Slider(2000, 15000, value=10000, step=500, label="Low-pass Filter (Hz)")

            # Run
            btn = gr.Button("Run Dubbing Pipeline", variant="primary", size="lg")

        # ── RIGHT: Output ───────────────────────
        with gr.Column(scale=6):

            # Console
            with gr.Column(elem_classes="card delay-4"):
                gr.HTML("""<div class="console-header"><span class="console-icon-wrap"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg></span> Pipeline Console</div>""")
                status_out = gr.Textbox(lines=14, interactive=False, elem_id="terminal-log", show_label=False, placeholder="Waiting for pipeline to start...")

            # Video
            with gr.Column(elem_classes="card delay-5"):
                gr.HTML("""<div class="card-title"><span class="card-icon-wrap"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/></svg></span> Final Render</div>""")
                vid_out = gr.Video(interactive=False, show_label=False, elem_classes="render-box")

    # ═══ FOOTER & CREDITS ═══════════════════════
    gr.HTML("""
    <div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 40px 0 20px; font-weight: 500; letter-spacing: 0.05em; z-index: 10; position: relative;">
        <span style="opacity: 0.6;">Designed & Engineered by</span><br>
        <div style="margin-top: 8px;">
            <span style="color: #A78BFA; font-weight: 800; font-size: 1rem; text-shadow: 0 0 15px rgba(167,139,250,0.4); letter-spacing: 0.08em;">CHENNOUFI DJEBRIL</span> 
            <span style="opacity: 0.4; margin: 0 12px; font-size: 0.9rem;">&times;</span> 
            <span style="color: #34D399; font-weight: 800; font-size: 1rem; text-shadow: 0 0 15px rgba(52,211,153,0.4); letter-spacing: 0.08em;">ZAIR HOCINE</span>
        </div>
    </div>
    """)

    # ═══ EVENTS ═════════════════════════════════
    btn.click(
        fn=process_pipeline,
        inputs=[yt_url, vid_in, target_lang, burn_subs, hf_token,
                whisper_choice, xtts_temp, atempo_max, sil_floor, min_slot,
                normalize_audio, target_volume, noise_reduction, high_pass, low_pass],
        outputs=[status_out, vid_out]
    )

    def update_lang_defaults(lang_name):
        d = LANGUAGE_MAP[lang_name]
        return d["whisper"], d["temperature"], d["atempo"], d["silence_floor"], d["min_slot"], d["normalize"], d["vol_target"], d["nr_level"], d["hp_freq"], d["lp_freq"]

    target_lang.change(fn=update_lang_defaults, inputs=[target_lang],
                       outputs=[whisper_choice, xtts_temp, atempo_max, sil_floor, min_slot, normalize_audio, target_volume, noise_reduction, high_pass, low_pass])

    def update_yt_preview(url):
        # Always clear vid_in path to prevent pipeline using old uploaded file
        if not url:
            return gr.update(visible=False, value=""), gr.update(visible=False, value=None)
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        if match:
            vid = match.group(1)
            iframe = f'<div style="animation: slideDown 0.4s ease-out; margin-top: 10px;"><iframe width="100%" height="200" src="https://www.youtube.com/embed/{vid}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="border-radius: 10px; border: 1px solid #1E1B2E;"></iframe></div>'
            return gr.update(visible=True, value=iframe), gr.update(visible=False, value=None)
        return gr.update(visible=False, value=""), gr.update(visible=False, value=None)

    yt_url.change(fn=update_yt_preview, inputs=[yt_url], outputs=[yt_preview, vid_in])

    def handle_upload(filepath):
        if filepath is None:
            return gr.update(visible=False, value=None), gr.update(value="")
        # Ensure we extract the string path from the temporary file object that Gradio > 3.0 sends via UploadButton
        val = getattr(filepath, "name", filepath)
        # return the string path to vid_in, clear yt_url so we don't accidentally dub from youtube
        return gr.update(visible=True, value=val), gr.update(value="")
    
    upload_btn.upload(fn=handle_upload, inputs=[upload_btn], outputs=[vid_in, yt_url])

if __name__ == "__main__":
    ui.queue().launch(server_name="0.0.0.0", server_port=7860,share=True)
