import os, argparse, json, numpy as np, torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import evaluate

def load_data(manifests_dir):
    ds = load_dataset("json", data_files={"test": os.path.join(manifests_dir, "test.jsonl")})
    ds = ds.cast_column("path", Audio(sampling_rate=16000))
    return ds["test"]

def transcribe_batch(batch, processor, model):
    audio = batch["path"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
    input_features = inputs.input_features
    with torch.no_grad():
        pred_ids = model.generate(input_features=input_features.to(model.device), max_length=225)
    preds = processor.batch_decode(pred_ids, skip_special_tokens=True)
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifests_dir', required=True)
    ap.add_argument('--models', nargs='+', required=True, help='List of model dirs or HF ids (first can be baseline)')
    ap.add_argument('--language', default='en')
    ap.add_argument('--batch_size', type=int, default=8)
    args = ap.parse_args()

    test = load_data(args.manifests_dir)
    wer_metric = evaluate.load("wer")

    results = {}
    for m in args.models:
        print(f"Evaluating {m} ...")
        processor = WhisperProcessor.from_pretrained(m, language=args.language, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(m).to("cuda" if torch.cuda.is_available() else "cpu")
        model.generation_config.language = args.language
        model.generation_config.task = "transcribe"
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")

        preds = []
        refs = []
        # simple batched loop
        for i in range(0, len(test), args.batch_size):
            batch = test[i:i+args.batch_size]
            p = transcribe_batch(batch, processor, model)
            preds.extend(p)
            refs.extend(batch["text"])

        wer = wer_metric.compute(predictions=preds, references=refs)
        results[m] = float(wer)
        print(m, "WER:", wer)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/wer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n== Summary ==")
    for k,v in results.items():
        print(f"{k}: WER={v:.4f}")

if __name__ == '__main__':
    main()
