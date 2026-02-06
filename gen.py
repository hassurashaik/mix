from pathlib import Path
import glob
import pandas as pd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from scipy.io.wavfile import write as wavwrite
from tqdm import tqdm
import argparse

# WSJ0 original sampling rate (DO NOT CHANGE)
FS_ORIG = 16000

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--wsj0_path",
    default=r"/content/wsj0-2mix"
    help="Path to WSJ0 root"
)
parser.add_argument("--output_folder", default="wsj0-mix")
parser.add_argument("--n_src", default=2, type=int)

# ✅ CHANGE TO 8 kHz (Conv-TasNet standard)
parser.add_argument("--samplerate", default=8000, type=int)

# ONLY MIN MODE
parser.add_argument("--len_mode", nargs="+", default=["min"])
args = parser.parse_args()

print("====================================")
print("WSJ0 path      :", args.wsj0_path)
print("Output folder  :", args.output_folder)
print("Sample rate    :", args.samplerate)
print("====================================")

# =========================
# READ ACTIVLEV
# =========================
activlev_df = pd.concat([
    pd.read_csv(f, sep=" ", names=["utt", "alev"])
    for f in glob.glob("metadata/activlev/*.txt")
])
activlev = dict(zip(activlev_df.utt, activlev_df.alev))

# =========================
# SAFE WAV WRITE (WINDOWS)
# =========================
def safe_write(path, audio, sr):
    audio = np.asarray(audio, dtype=np.float32)

    # Normalize safely
    maxv = np.max(np.abs(audio))
    if maxv > 0:
        audio = audio / maxv * 0.9

    audio = np.clip(audio, -1.0, 1.0)

    # Convert to int16
    audio_int16 = (audio * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)

    # SciPy WAV writer (robust on Windows)
    wavwrite(str(path), sr, audio_int16)

# =========================
# MAIN LOOP
# =========================
for split in ["tr", "cv", "tt"]:
    base = (
        Path(args.output_folder)
        / f"{args.n_src}speakers"
        / f"wav{args.samplerate // 1000}k"
    )

    mix_dir = base / "min" / split / "mix"
    src_dirs = [base / "min" / split / f"s{i+1}" for i in range(args.n_src)]

    mix_dir.mkdir(parents=True, exist_ok=True)
    for d in src_dirs:
        d.mkdir(parents=True, exist_ok=True)

    mix_df = pd.read_csv(
        f"metadata/mix_{args.n_src}_spk_{split}.txt",
        sep=" ",
        header=None
    )

    print(f"\nGenerating {split} set...")

    for idx, row in tqdm(mix_df.iterrows(), total=len(mix_df)):
        src_paths = [Path(args.wsj0_path) / row[i * 2] for i in range(args.n_src)]
        snrs = [row[i * 2 + 1] for i in range(args.n_src)]

        sources = []
        for p in src_paths:
            s, _ = sf.read(p, dtype="float32")
            # 🔽 Resample 16k → 8k
            s = resample_poly(s, args.samplerate, FS_ORIG)
            sources.append(s)

        min_len = min(len(s) for s in sources)
        sources = [s[:min_len] for s in sources]

        scaled = []
        for s, snr, p in zip(sources, snrs, src_paths):
            scale = activlev[p.stem]
            scaled.append(s / np.sqrt(scale) * 10 ** (snr / 20))

        stacked = np.stack(scaled)
        mix = stacked.sum(axis=0)

        # Short, safe filename
        name = f"{split}_{idx:06d}.wav"

        g = max(1.0, np.max(np.abs(mix))) / 0.9

        safe_write(mix_dir / name, mix / g, args.samplerate)
        for i, s in enumerate(stacked):
            safe_write(src_dirs[i] / name, s / g, args.samplerate)

print("\n✅ WSJ0-mix 8k generation finished SUCCESSFULLY")
