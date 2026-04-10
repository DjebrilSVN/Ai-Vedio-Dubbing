from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import re
import torch
import warnings
import gc  
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")

# ==========================================
#  PRE-DOWNLOAD DIRECTLY TO DISK ON BOOT
# ==========================================
print(" Checking/Downloading NLLB-1.3B to disk (This may take a while)...")
model_name = "facebook/nllb-200-distilled-1.3B"

# This ONLY downloads the files to your hard drive. It does NOT use any RAM!
snapshot_download(repo_id=model_name)
print(" NLLB Model is fully downloaded and securely cached on your hard drive!")

app = FastAPI()

class TranslationRequest(BaseModel):
    transcript_filename: str

@app.post("/translate")
def run_translation(request: TranslationRequest):
    print(f" Service B activated! Translating: {request.transcript_filename}")
    
    # --- SHARED FOLDER PATHS ---
    data_dir = "/app/data"
    input_filepath = f"{data_dir}/{request.transcript_filename}"
    output_filename = f"{data_dir}/translated_transcript_fixed.json"

    if not os.path.exists(input_filepath):
        raise HTTPException(status_code=404, detail=f"File {input_filepath} not found.")

    # ==========================================
    # 🚀 LOAD MODEL FROM HARD DRIVE -> GPU VRAM
    # ==========================================
    print("Loading NLLB-200-1.3B into GPU memory...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        revision="main", 
        local_files_only=False 
    ).to("cuda")
    french_id = tokenizer.convert_tokens_to_ids("fra_Latn")
    print(" NLLB Model loaded into VRAM!")

    # ==========================================
    # PHASE 1: LOAD DATA
    # ==========================================
    with open(input_filepath, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    for entry in transcript_data:
        if "speaker" not in entry:
            entry["speaker"] = "UNKNOWN_SPEAKER"

    # ==========================================
    # PHASE 2: OPTIMIZED TEMPORAL GROUPING
    # ==========================================
    print("Grouping dialogue into bite-sized cinematic chunks...")
    MAX_PAUSE_SECONDS = 0.4  
    MAX_CHARS = 100          
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
        
        must_cut = (
            speaker_changed or 
            (is_end_of_sentence and len(text) > MIN_CHARS) or 
            time_gap > MAX_PAUSE_SECONDS or 
            len(text) > MAX_CHARS
        )
        
        if not must_cut:
            current_sentence["text"] += " " + word_data["word"]
            current_sentence["end"] = word_data["end"] 
        else:
            grouped_sentences.append(current_sentence)
            current_sentence = {
                "speaker": word_data["speaker"], 
                "text": word_data["word"], 
                "start": word_data["start"], 
                "end": word_data["end"]
            }

    grouped_sentences.append(current_sentence)

    # ==========================================
    # PHASE 3: DEEP TRANSLATION
    # ==========================================
    print("Processing Deep Translation via Beam Search...")
    translated_data = []

    for block in grouped_sentences:
        original_text = block["text"].strip()
        clean_text = original_text.replace("♪", "").strip()
        if not clean_text: continue
            
        inputs = tokenizer(clean_text, return_tensors="pt").to("cuda")
        
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=french_id, 
            max_length=100, 
            num_beams=5,             
            repetition_penalty=1.2   
        )
        
        french_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        print(f"[{block['start']}s] {block['speaker']}: {french_text}")
        
        translated_data.append({
            "speaker": block["speaker"],
            "start": block["start"],
            "end": block["end"],
            "english_original": original_text,
            "french_translation": french_text
        })

    # ==========================================
    # PHASE 4: FINAL EXPORT
    # ==========================================
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(translated_data, f, indent=4, ensure_ascii=False)

    print(f" Translation Complete. Saved to {output_filename}")

    # ==========================================
    #  THE VRAM ASSASSINATION
    # ==========================================
    print(" Wiping NLLB from GPU memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {"status": "success", "file_ready": "translated_transcript_fixed.json"}