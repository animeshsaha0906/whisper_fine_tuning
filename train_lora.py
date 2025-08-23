import os, argparse, json, random, numpy as np, torch
from datasets import load_dataset, Audio
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_datasets(manifests_dir):
    data_files = {
        "train": os.path.join(manifests_dir, "train.jsonl"),
        "test": os.path.join(manifests_dir, "test.jsonl"),
    }
    ds = load_dataset("json", data_files=data_files)
    ds = ds.cast_column("path", Audio(sampling_rate=16000))
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifests_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--base_model', default='openai/whisper-base')
    ap.add_argument('--language', default='en')
    ap.add_argument('--num_train_epochs', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=2e-4)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--grad_accum', type=int, default=2)
    ap.add_argument('--warmup_ratio', type=float, default=0.1)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--freeze_encoder', action='store_true')
    ap.add_argument('--spec_augment', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ds = prepare_datasets(args.manifests_dir)

    processor = WhisperProcessor.from_pretrained(args.base_model, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False

    # PEFT LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "k_proj","q_proj","v_proj","out_proj","fc1","fc2"
        ]
    )
    model = get_peft_model(model, lora_config)

    # feature extraction / tokenization
    def preprocess(batch):
        audio = batch["path"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with processor.as_target_processor():
            labels = processor(batch["text"]).input_ids
        batch["input_features"] = inputs.input_features[0]
        batch["labels"] = labels
        return batch

    cols = ds["train"].column_names
    ds_proc = ds.map(preprocess, remove_columns=cols, num_proc=1)

    # optional SpecAugment (time/freq masking) via torch native API
    if args.spec_augment:
        import torchaudio.transforms as T
        specaug = torch.nn.Sequential(
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=50),
        )
        def collate_fn(features):
            input_features = torch.stack([f["input_features"] for f in features])
            # apply SpecAugment on log-mel (B,80,T)
            input_features = specaug(input_features)
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            return {"input_features": input_features, "labels": labels}
    else:
        collate_fn = DataCollatorForSeq2Seq(processor.tokenizer, model=model, label_pad_token_id=-100)

    wer = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        # replace -100 in the labels as we can't decode them
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=pred_str, references=label_str)}

    last_ckpt = None
    if os.path.isdir(args.output_dir):
        last_ckpt = get_last_checkpoint(args.output_dir)

    tr_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        bf16=False,
        generation_max_length=225,
        report_to=["none"],
        save_total_limit=2,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=tr_args,
        train_dataset=ds_proc["train"],
        eval_dataset=ds_proc["test"],
        data_collator=collate_fn,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model()
    processor.save_pretrained(args.output_dir)

    # Save final eval
    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)

if __name__ == "__main__":
    main()
