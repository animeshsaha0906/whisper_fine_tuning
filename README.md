# 2025 UG Test - ASR Model (Starter Repo)



- `prepare_data.py`: segment audio using timestamps and build HF-ready manifests
- `train_lora.py`: LoRA fine-tuning of Whisper-base
- `train_full.py`: selective fine-tuning (decoder-only or last-N blocks)
- `eval.py`: evaluate WER on test set and compare to baseline Whisper-base
- `report_template.md`: 2-page report template

---

## 0) Environment

```bash
conda create -n asr python=3.10 -y
conda activate asr
pip install -r requirements.txt
```



## 1) Data Preparation

Expected inputs:
- **Audio**: One or more WAV/FLAC files with children speech (16 kHz preferred; script will resample).
- **Transcripts with timestamps**: CSV/TSV/JSONL containing at least: `audio_path`, `start`, `end`, `text`.
  - Times are in seconds (float ok).

Example CSV header:
```text
audio_path,start,end,text,split
/abs/path/child1.wav,0.53,1.92,hello there,train
/abs/path/child1.wav,2.05,3.10,how are you,test
```


Run segmentation & manifest build:
```bash
python prepare_data.py       --table /path/to/transcripts.csv       --out_dir data       --format csv       --sample_rate 16000       --min_duration 0.6       --max_duration 20.0       --trim_silence       --normalize_text
```

Output:
- `data/segments/train/*.wav`
- `data/segments/test/*.wav`
- `data/manifests/train.jsonl` and `data/manifests/test.jsonl` (HF `datasets` compatible)

## 2) LoRA Fine-tuning

```bash
accelerate config  # run once to set up (defaults are fine)

accelerate launch train_lora.py       --manifests_dir data/manifests       --output_dir outputs/lora-whisper-base       --base_model openai/whisper-base       --language en       --num_train_epochs 5       --learning_rate 2e-4       --batch_size 8       --grad_accum 2       --warmup_ratio 0.1       --lora_r 16       --lora_alpha 32       --lora_dropout 0.05       --freeze_encoder        --spec_augment
```

Notes:
- Start with **encoder frozen** for stability on small datasets.
- `spec_augment` and optional speed perturbation help with children's speech variability.

## 3) Selective Fine-tuning (decoder-only / last-N)

```bash
accelerate launch train_full.py       --manifests_dir data/manifests       --output_dir outputs/decoder-only       --base_model openai/whisper-base       --language en       --tune_mode decoder       --num_train_epochs 5       --learning_rate 1e-5       --batch_size 8       --grad_accum 2       --warmup_ratio 0.1       --spec_augment
```

Or last-N blocks (encoder+decoder top layers):
```bash
accelerate launch train_full.py       --manifests_dir data/manifests       --output_dir outputs/last2       --base_model openai/whisper-base       --language en       --tune_mode last_n       --last_n 2       --num_train_epochs 5       --learning_rate 5e-6       --batch_size 8       --grad_accum 4       --warmup_ratio 0.1       --spec_augment
```

## 4) Evaluation (WER)

Compare multiple checkpoints against the **original** Whisper-base:
```bash
python eval.py       --manifests_dir data/manifests       --models openai/whisper-base outputs/lora-whisper-base outputs/decoder-only outputs/last2       --language en       --batch_size 8
```


---
**Subject title for submission:** `2025 UG Test - ASR Model`

Fill `report_template.md`, zip your repo, and include a link to model + code.
