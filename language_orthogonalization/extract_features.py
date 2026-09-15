"""Extracts layer-wise S3M embeddings.npy + index.csv for one backbone.

Saves every hidden state from output_hidden_states=True, including layer 0
(the CNN feature-encoder output), mean+std pooled per utterance. This is
the (L, N, D) input pooling.load_features expects.

ECAPA-TDNN and the VoxLingua107 LID embedding are not produced by this
script; extract those with SpeechBrain directly.

Usage
-----
    python -m language_orthogonalization.extract_features hubert_large
    python -m language_orthogonalization.extract_features all
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

MANIFEST  = Path(os.environ.get("LOPD_MANIFEST",  "./manifests/manifest.csv"))
FEAT_ROOT = Path(os.environ.get("LOPD_FEAT_ROOT", "./features"))
SR = 16000
CHUNK_SAMPLES = SR * 30
NORM_TARGET_DB = -23.0

MODELS = {
    "hubert_large":    {"hf_id": "facebook/hubert-large-ll60k",     "kind": "wav2vec2", "dim": 1024},
    "wavlm":           {"hf_id": "microsoft/wavlm-large",           "kind": "wavlm",    "dim": 1024},
    "xls_r_300m":      {"hf_id": "facebook/wav2vec2-xls-r-300m",    "kind": "wav2vec2", "dim": 1024},
    "mms_300m":        {"hf_id": "facebook/mms-300m",               "kind": "wav2vec2", "dim": 1024},
    "w2v2_large_lv60": {"hf_id": "facebook/wav2vec2-large-lv60",    "kind": "wav2vec2", "dim": 1024},
    "whisper_large":   {"hf_id": "openai/whisper-large-v3",         "kind": "whisper",  "dim": 1280},
    "ast_audioset":    {"hf_id": "MIT/ast-finetuned-audioset-10-10-0.4593", "kind": "ast", "dim": 768},
}


def load_audio(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float32)
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-12))
    if rms > 1e-8:
        target = 10 ** (NORM_TARGET_DB / 20.0)
        wav = wav * (target / rms)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak > 0.99:
            wav = wav * (0.99 / peak)
    return wav


def chunks_of(wav: np.ndarray, n: int) -> list[np.ndarray]:
    if len(wav) <= n:
        return [wav]
    return [wav[i:i + n] for i in range(0, len(wav), n)]


@torch.inference_mode()
def forward_wav2vec_like(model, processor, device, wav: np.ndarray) -> list[np.ndarray]:
    layer_parts: list[list[np.ndarray]] = []
    for chunk in chunks_of(wav, CHUNK_SAMPLES):
        if len(chunk) < SR // 50:
            chunk = np.pad(chunk, (0, SR // 50 - len(chunk)))
        inputs = processor(chunk, sampling_rate=SR, return_tensors="pt")
        iv = inputs.input_values.to(device).to(model.dtype)
        am = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
        out = model(iv, attention_mask=am, output_hidden_states=True)
        hs = out.hidden_states
        if not layer_parts:
            layer_parts = [[] for _ in hs]
        for li, h in enumerate(hs):
            layer_parts[li].append(h.squeeze(0).float().cpu().numpy())
    return [np.concatenate(parts, axis=0) for parts in layer_parts]


@torch.inference_mode()
def forward_whisper(model, processor, device, wav: np.ndarray) -> list[np.ndarray]:
    layer_parts: list[list[np.ndarray]] = []
    for chunk in chunks_of(wav, CHUNK_SAMPLES):
        inputs = processor(chunk, sampling_rate=SR, return_tensors="pt")
        feats = inputs.input_features.to(device).to(model.dtype)
        out = model.encoder(feats, output_hidden_states=True)
        hs = out.hidden_states
        if not layer_parts:
            layer_parts = [[] for _ in hs]
        for li, h in enumerate(hs):
            layer_parts[li].append(h.squeeze(0).float().cpu().numpy())
    return [np.concatenate(parts, axis=0) for parts in layer_parts]


@torch.inference_mode()
def forward_ast(model, processor, device, wav: np.ndarray) -> list[np.ndarray]:
    AST_CHUNK = SR * 10
    chunks = [wav[i:i + AST_CHUNK] for i in range(0, len(wav), AST_CHUNK)] or [wav]
    layer_parts: list[list[np.ndarray]] = []
    for chunk in chunks:
        if len(chunk) < SR // 4:
            chunk = np.pad(chunk, (0, SR // 4 - len(chunk)))
        inputs = processor(chunk, sampling_rate=SR, return_tensors="pt")
        iv = inputs.input_values.to(device).to(model.dtype)
        out = model(iv, output_hidden_states=True)
        hs = out.hidden_states
        if not layer_parts:
            layer_parts = [[] for _ in hs]
        for li, h in enumerate(hs):
            layer_parts[li].append(h.squeeze(0).float().cpu().numpy())
    return [np.concatenate(parts, axis=0) for parts in layer_parts]


def pool(h: np.ndarray) -> np.ndarray:
    return np.concatenate([h.mean(0), h.std(0)]).astype(np.float16)


def run_one(model_key: str, manifest: pd.DataFrame) -> None:
    cfg = MODELS[model_key]
    out_dir = FEAT_ROOT / (model_key + "_layers")
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_path = out_dir / "embeddings.npy"
    idx_path = out_dir / "index.csv"
    if emb_path.exists():
        print(f"[skip] {out_dir} exists")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    from transformers import AutoFeatureExtractor
    if cfg["kind"] in ("wav2vec2", "wavlm"):
        from transformers import AutoModel
        processor = AutoFeatureExtractor.from_pretrained(cfg["hf_id"])
        model = AutoModel.from_pretrained(cfg["hf_id"], torch_dtype=dtype).to(device).eval()
        fwd = forward_wav2vec_like
    elif cfg["kind"] == "whisper":
        from transformers import WhisperFeatureExtractor, WhisperModel
        processor = WhisperFeatureExtractor.from_pretrained(cfg["hf_id"])
        model = WhisperModel.from_pretrained(cfg["hf_id"], torch_dtype=dtype).to(device).eval()
        fwd = forward_whisper
    else:
        from transformers import ASTFeatureExtractor, ASTModel
        processor = ASTFeatureExtractor.from_pretrained(cfg["hf_id"])
        model = ASTModel.from_pretrained(cfg["hf_id"], torch_dtype=dtype).to(device).eval()
        fwd = forward_ast

    print(f"[load] {model_key} <- {cfg['hf_id']} on {device}", flush=True)

    n = len(manifest)
    pooled_dim = 2 * cfg["dim"]
    out = None
    paths = manifest["path"].tolist()

    t0 = time.time()
    for i, p in enumerate(paths):
        wav = load_audio(p)
        hs_list = fwd(model, processor, device, wav)
        if out is None:
            n_layers = len(hs_list)
            print(f"[info] {model_key}: {n_layers} layers, allocating ({n_layers},{n},{pooled_dim}) fp16 "
                  f"= {n_layers*n*pooled_dim*2/1e9:.2f} GB", flush=True)
            out = np.zeros((n_layers, n, pooled_dim), dtype=np.float16)
        for li, h in enumerate(hs_list):
            out[li, i] = pool(h)
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  [{model_key}] {i+1:4d}/{n}  ({rate:.1f} utt/s, ETA {eta/60:.1f} min)", flush=True)

    np.save(emb_path, out)
    idx = manifest[["path", "speaker_id", "lang", "cohort", "group", "task"]].copy()
    idx.to_csv(idx_path, index=False)
    print(f"[done] {model_key}_layers: {out.shape}  {out.nbytes/1e6:.1f} MB")
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(MODELS.keys()) + ["all"])
    args = ap.parse_args()

    manifest = pd.read_csv(MANIFEST).sort_values("path").reset_index(drop=True)
    print(f"[info] {len(manifest)} utterances")
    targets = list(MODELS.keys()) if args.model == "all" else [args.model]
    for k in targets:
        run_one(k, manifest)


if __name__ == "__main__":
    main()
