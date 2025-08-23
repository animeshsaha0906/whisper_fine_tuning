import os, argparse, pandas as pd, json, math
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
from utils import basic_normalize, strip_punct

def load_table(path, fmt):
    if fmt == 'csv':
        return pd.read_csv(path)
    elif fmt == 'tsv':
        return pd.read_csv(path, sep='\t')
    elif fmt == 'jsonl':
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    else:
        raise ValueError('format must be csv/tsv/jsonl')

def trim_silence_waveform(y, sr, top_db=30, pad_ms=50):
    # trims leading/trailing silence using librosa; returns trimmed y
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    pad = int(sr * pad_ms / 1000.0)
    if pad > 0:
        yt = np.pad(yt, (pad, pad), mode='constant')
    return yt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True, help='CSV/TSV/JSONL with columns: audio_path,start,end,text[,split]')
    ap.add_argument('--format', choices=['csv','tsv','jsonl'], required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--sample_rate', type=int, default=16000)
    ap.add_argument('--min_duration', type=float, default=0.6)
    ap.add_argument('--max_duration', type=float, default=20.0)
    ap.add_argument('--trim_silence', action='store_true')
    ap.add_argument('--normalize_text', action='store_true')
    ap.add_argument('--concat_short_threshold', type=float, default=0.0, help='If >0, concatenate adjacent same-file segments shorter than this (seconds).')
    ap.add_argument('--default_split', choices=['train','test'], default=None, help='Used if table has no split column.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seg_dir = os.path.join(args.out_dir, 'segments')
    mani_dir = os.path.join(args.out_dir, 'manifests')
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(mani_dir, exist_ok=True)

    df = load_table(args.table, args.format)
    needed_cols = {'audio_path','start','end','text'}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f'Table must have columns: {needed_cols}')
    if 'split' not in df.columns and args.default_split is None:
        raise ValueError('Table missing split column; provide --default_split')
    if 'split' not in df.columns:
        df['split'] = args.default_split

    # optional concatenation of very short adjacent segments
    if args.concat_short_threshold > 0:
        new_rows = []
        for audio_path, g in df.sort_values(['audio_path','start']).groupby('audio_path'):
            buffer = None
            for _, r in g.iterrows():
                dur = float(r['end']) - float(r['start'])
                if buffer is None:
                    buffer = r.to_dict()
                    continue
                prev_dur = float(buffer['end']) - float(buffer['start'])
                # if both short and contiguous (<= 0.2s gap), merge
                if prev_dur < args.concat_short_threshold and dur < args.concat_short_threshold and abs(float(r['start']) - float(buffer['end'])) <= 0.2 and r['split']==buffer['split']:
                    buffer['end'] = float(r['end'])
                    buffer['text'] = str(buffer['text']).strip() + ' ' + str(r['text']).strip()
                else:
                    new_rows.append(buffer)
                    buffer = r.to_dict()
            if buffer is not None:
                new_rows.append(buffer)
        df = pd.DataFrame(new_rows)

    writers = {
        'train': open(os.path.join(mani_dir, 'train.jsonl'), 'w', encoding='utf-8'),
        'test': open(os.path.join(mani_dir, 'test.jsonl'), 'w', encoding='utf-8'),
    }
    try:
        counters = {'train':0,'test':0}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            audio_path = row['audio_path']
            start = float(row['start'])
            end = float(row['end'])
            split = row['split']
            text = str(row['text'])

            if end <= start: 
                continue
            duration = end - start
            if duration < args.min_duration or duration > args.max_duration:
                continue

            # load slice, resample if needed
            y, sr = sf.read(audio_path, always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            s = int(start * sr)
            e = int(end * sr)
            y = y[max(0,s):min(len(y), e)]

            if sr != args.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.sample_rate)
                sr = args.sample_rate

            if args.trim_silence:
                y = trim_silence_waveform(y, sr, top_db=30, pad_ms=30)

            if len(y) < int(args.min_duration * sr):
                continue

            # normalize peak to 0.9 (simple)
            peak = np.max(np.abs(y)) + 1e-9
            y = 0.9 * y / peak

            # text normalization (lightweight)
            if args.normalize_text:
                text = basic_normalize(text)

            out_split_dir = os.path.join(seg_dir, split)
            os.makedirs(out_split_dir, exist_ok=True)
            uid = f"{counters[split]:06d}"
            out_wav = os.path.join(out_split_dir, f"{uid}.wav")
            sf.write(out_wav, y, sr)

            # write manifest
            item = {"path": os.path.abspath(out_wav), "text": text, "duration": len(y)/sr}
            writers[split].write(json.dumps(item, ensure_ascii=False) + "\n")
            counters[split]+=1
    finally:
        for w in writers.values():
            w.close()

    print("Done. Manifests at:", mani_dir)

if __name__ == '__main__':
    main()
