import os
import json
import time
import shutil
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn import functional as F


# =========================================================
# 기본 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = BASE_DIR / "datasets"
SOURCE_TEXTS_DIR = DATASETS_DIR / "source_texts"

WORK_DIR = BASE_DIR / "work"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
LOG_DIR = BASE_DIR / "logs"

MANIFEST_PATH = WORK_DIR / "source_manifest.json"
MERGED_TEXT_PATH = WORK_DIR / "merged_corpus.txt"
TOKENIZED_IDS_PATH = WORK_DIR / "token_ids_uint32.npy"
TRAIN_LOG_PATH = LOG_DIR / "train_log.txt"

SPM_PREFIX = TOKENIZER_DIR / "spm_korean_chat"
SPM_MODEL_PATH = TOKENIZER_DIR / "spm_korean_chat.model"
SPM_VOCAB_PATH = TOKENIZER_DIR / "spm_korean_chat.vocab"

LATEST_CKPT_PATH = CHECKPOINT_DIR / "latest_checkpoint.pth"
BEST_CKPT_PATH = CHECKPOINT_DIR / "best_checkpoint.pth"


# =========================================================
# 토크나이저 설정
# =========================================================
VOCAB_SIZE = 100
CHARACTER_COVERAGE = 0.9995
TOKENIZER_MODEL_TYPE = "bpe"
REBUILD_TOKENIZER_IF_MISSING = True
FORCE_REBUILD_TOKENIZER = False


# =========================================================
# 학습 설정
# =========================================================
BATCH_SIZE = 8
BLOCK_SIZE = 5
MAX_NEW_ITERS_PER_RUN = 3000
EVAL_INTERVAL = 200
SAVE_INTERVAL = 200
EVAL_ITERS = 50

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
TRAIN_SPLIT = 0.9

N_EMBD = 256
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()
USE_COMPILE = hasattr(torch, "compile")


# =========================================================
# 유틸
# =========================================================
def ensure_dirs() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg)
    with open(TRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except Exception:
            pass
    os.replace(tmp, path)


def safe_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except Exception:
            pass
    os.replace(tmp, path)


def safe_torch_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copy2(path, bak)
        except Exception:
            pass
    os.replace(tmp, path)


# =========================================================
# 소스 txt 병합
# =========================================================
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"files": {}, "order": []}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}, "order": []}


def discover_source_txt_files() -> List[Path]:
    files = []
    for path in SOURCE_TEXTS_DIR.rglob("*.txt"):
        if path.is_file():
            files.append(path)
    files.sort(key=lambda p: str(p).lower())
    return files


def update_merged_corpus_append_only() -> Tuple[bool, List[str]]:
    manifest = load_manifest()
    old_files = manifest.get("files", {})

    current_files = discover_source_txt_files()
    current_info = {}
    changed_existing = False
    new_added = []
    logs = []

    for path in current_files:
        rel = str(path.relative_to(SOURCE_TEXTS_DIR))
        file_hash = sha256_file(path)
        stat = path.stat()

        current_info[rel] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": file_hash,
        }

        if rel not in old_files:
            new_added.append(rel)
        else:
            if old_files[rel].get("sha256") != file_hash:
                changed_existing = True

    removed_files = [rel for rel in old_files.keys() if rel not in current_info]
    if removed_files:
        changed_existing = True

    if not MERGED_TEXT_PATH.exists():
        changed_existing = True

    if changed_existing:
        logs.append("기존 txt 변경/삭제 감지. merged_corpus 전체 재구성.")
        parts = []
        for path in current_files:
            rel = str(path.relative_to(SOURCE_TEXTS_DIR))
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            parts.append(f"[문서 시작:{rel}]\n{text}\n[문서 끝:{rel}]")
        merged_text = "\n\n".join(parts).strip() + "\n"
        safe_write_text(MERGED_TEXT_PATH, merged_text)

    elif new_added:
        logs.append(f"새 txt {len(new_added)}개 감지. merged_corpus 뒤에 추가.")
        with open(MERGED_TEXT_PATH, "a", encoding="utf-8") as out:
            for rel in new_added:
                path = SOURCE_TEXTS_DIR / rel
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue
                out.write(f"\n\n[문서 시작:{rel}]\n{text}\n[문서 끝:{rel}]\n")
    else:
        logs.append("새 txt 없음. merged_corpus 유지.")

    new_manifest = {
        "files": current_info,
        "order": [str(p.relative_to(SOURCE_TEXTS_DIR)) for p in current_files],
    }
    safe_write_json(MANIFEST_PATH, new_manifest)

    corpus_changed = changed_existing or bool(new_added) or (not TOKENIZED_IDS_PATH.exists())
    return corpus_changed, logs


# =========================================================
# SentencePiece 토크나이저
# =========================================================
def build_sentencepiece_tokenizer_if_needed() -> None:
    if FORCE_REBUILD_TOKENIZER:
        log("FORCE_REBUILD_TOKENIZER=True -> 토크나이저 재학습")
    elif SPM_MODEL_PATH.exists() and SPM_VOCAB_PATH.exists():
        log("기존 SentencePiece 토크나이저 재사용")
        return
    elif not REBUILD_TOKENIZER_IF_MISSING:
        raise FileNotFoundError("토크나이저가 없는데 자동 재생성을 비활성화했다.")

    if not MERGED_TEXT_PATH.exists():
        raise FileNotFoundError(f"merged corpus 없음: {MERGED_TEXT_PATH}")

    log("SentencePiece 토크나이저 학습 시작")

    spm.SentencePieceTrainer.Train(
        input=str(MERGED_TEXT_PATH),
        model_prefix=str(SPM_PREFIX),
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        model_type=TOKENIZER_MODEL_TYPE,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )

    log("SentencePiece 토크나이저 학습 완료")


def load_sp_processor() -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    ok = sp.load(str(SPM_MODEL_PATH))
    if not ok:
        raise RuntimeError("SentencePiece 모델 로드 실패")
    return sp


def tokenize_corpus_to_memmap(force_rebuild: bool = False) -> Tuple[np.memmap, spm.SentencePieceProcessor]:
    build_sentencepiece_tokenizer_if_needed()
    sp = load_sp_processor()

    if TOKENIZED_IDS_PATH.exists() and not force_rebuild:
        arr = np.load(TOKENIZED_IDS_PATH, mmap_mode="r")
        log(f"기존 token ids 캐시 재사용 | 길이={len(arr):,}")
        return arr, sp

    log("토큰화 캐시 생성 시작")
    text = MERGED_TEXT_PATH.read_text(encoding="utf-8", errors="ignore")
    ids = sp.encode(text, out_type=int)
    arr = np.array(ids, dtype=np.uint32)

    mm = np.lib.format.open_memmap(TOKENIZED_IDS_PATH, mode="w+", dtype=np.uint32, shape=arr.shape)
    mm[:] = arr[:]
    del mm

    arr_read = np.load(TOKENIZED_IDS_PATH, mmap_mode="r")
    log(f"토큰화 캐시 생성 완료 | 길이={len(arr_read):,} | vocab={sp.vocab_size():,}")
    return arr_read, sp


# =========================================================
# 모델
# =========================================================
@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = BLOCK_SIZE
    n_embd: int = N_EMBD
    n_head: int = N_HEAD
    n_layer: int = N_LAYER
    dropout: float = DROPOUT


class Head(nn.Module):
    def __init__(self, head_size: int, config: ModelConfig):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config.n_head, head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B2, T2, C = logits.shape
            logits = logits.view(B2 * T2, C)
            targets = targets.view(B2 * T2)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


def create_model(vocab_size: int) -> GPTLanguageModel:
    config = ModelConfig(vocab_size=vocab_size)
    return GPTLanguageModel(config)


def maybe_compile_model(model: nn.Module) -> nn.Module:
    if DEVICE == "cuda" and USE_COMPILE:
        try:
            model = torch.compile(model)
            log("torch.compile 적용 완료")
        except Exception as e:
            log(f"torch.compile 생략: {e}")
    return model


# =========================================================
# 체크포인트
# =========================================================
def find_resume_checkpoint() -> Optional[Path]:
    if LATEST_CKPT_PATH.exists():
        return LATEST_CKPT_PATH
    ckpts = sorted(CHECKPOINT_DIR.glob("iter_*.pth"))
    if ckpts:
        return ckpts[-1]
    return None


def load_or_create_model_optimizer(vocab_size: int):
    resume_path = find_resume_checkpoint()

    model = create_model(vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    start_iter = 0
    best_val_loss = float("inf")

    if resume_path and resume_path.exists():
        log(f"체크포인트 로드: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)

        ckpt_config = ckpt.get("config", {})
        if ckpt_config.get("vocab_size") != vocab_size:
            raise ValueError(
                f"체크포인트 vocab_size={ckpt_config.get('vocab_size')} / 현재 tokenizer vocab_size={vocab_size}. "
                f"토크나이저를 고정 재사용해야 이어학습이 안전하다."
            )

        model = create_model(vocab_size).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=BETAS,
            weight_decay=WEIGHT_DECAY,
        )

        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            log("optimizer 상태 복원 완료")
        except Exception as e:
            log(f"optimizer 상태 복원 실패, 새 optimizer 사용: {e}")

        if USE_AMP and "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                log("GradScaler 상태 복원 완료")
            except Exception as e:
                log(f"GradScaler 복원 실패: {e}")

        start_iter = int(ckpt.get("iter_num", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    model = maybe_compile_model(model)
    return model, optimizer, scaler, start_iter, best_val_loss


def save_checkpoint(model, optimizer, scaler, iter_num, best_val_loss, extra_name=None):
    unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    ckpt = {
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "model_state_dict": unwrapped_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if USE_AMP else {},
        "config": asdict(unwrapped_model.config),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    safe_torch_save(ckpt, LATEST_CKPT_PATH)
    if extra_name:
        safe_torch_save(ckpt, CHECKPOINT_DIR / extra_name)


# =========================================================
# 데이터 배치
# =========================================================
def make_train_val_views(data: np.memmap):
    n = len(data)
    split_idx = int(n * TRAIN_SPLIT)
    return data[:split_idx], data[split_idx:]


def get_batch(data: np.ndarray, batch_size: int, block_size: int):
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("토큰 데이터가 block_size보다 너무 짧다.")

    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([torch.from_numpy(np.array(data[i:i + block_size], dtype=np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(np.array(data[i + 1:i + block_size + 1], dtype=np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    out = {}
    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split_data, BATCH_SIZE, BLOCK_SIZE)
            with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


# =========================================================
# 텍스트 생성
# =========================================================
@torch.no_grad()
def generate_text(model, sp, prompt="[대화 시작]\n화자1:", max_new_tokens=120):
    model.eval()
    ids = sp.encode(prompt, out_type=int)
    if not ids:
        ids = [sp.bos_id()] if sp.bos_id() >= 0 else [1]

    idx = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    eos_id = sp.eos_id()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id >= 0 and int(idx_next.item()) == eos_id:
            break

    return sp.decode(idx[0].tolist())


# =========================================================
# 메인
# =========================================================
def main():
    ensure_dirs()
    set_seed(SEED)

    log("=" * 60)
    log("ChatChosim AI 학습 시작")
    log(f"DEVICE = {DEVICE}")
    log(f"BASE_DIR = {BASE_DIR}")
    log(f"SOURCE_TEXTS_DIR = {SOURCE_TEXTS_DIR}")
    log(f"WORK_DIR = {WORK_DIR}")
    log(f"CHECKPOINT_DIR = {CHECKPOINT_DIR}")
    log(f"TOKENIZER = SentencePiece {TOKENIZER_MODEL_TYPE}, vocab_size={VOCAB_SIZE}")

    if not any(SOURCE_TEXTS_DIR.rglob("*.txt")):
        log("datasets/source_texts 폴더 안에 txt가 없다.")
        log("여기에 학습용 txt를 넣고 다시 실행하면 된다.")
        return

    corpus_changed, merge_logs = update_merged_corpus_append_only()
    for m in merge_logs:
        log(m)

    if not MERGED_TEXT_PATH.exists():
        log("merged_corpus.txt 생성 실패")
        return

    force_retokenize = corpus_changed or FORCE_REBUILD_TOKENIZER or (not TOKENIZED_IDS_PATH.exists())
    token_data, sp = tokenize_corpus_to_memmap(force_rebuild=force_retokenize)

    vocab_size = sp.vocab_size()
    if len(token_data) < BLOCK_SIZE + 10:
        log("토큰 데이터가 너무 짧아서 학습할 수 없다.")
        return

    train_data, val_data = make_train_val_views(token_data)
    log(f"token 길이 전체: {len(token_data):,}")
    log(f"train token 길이: {len(train_data):,}")
    log(f"val token 길이: {len(val_data):,}")

    model, optimizer, scaler, start_iter, best_val_loss = load_or_create_model_optimizer(vocab_size)
    log(f"이어학습 시작 iter = {start_iter:,}")
    log(f"best val loss = {best_val_loss:.6f}")

    iter_num = start_iter
    run_end_iter = start_iter + MAX_NEW_ITERS_PER_RUN
    t0 = time.time()
    model.train()

    while iter_num < run_end_iter:
        if iter_num % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            log(f"[iter {iter_num:,}] train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    extra_name="best_checkpoint.pth",
                )
                log("best checkpoint 저장 완료")

        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            _, loss = model(xb, yb)

        scaler.scale(loss).backward()

        if GRAD_CLIP is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        iter_num += 1

        if iter_num % 50 == 0:
            elapsed = time.time() - t0
            ips = (iter_num - start_iter) / elapsed if elapsed > 0 else 0.0
            log(f"iter {iter_num:,}/{run_end_iter:,} | loss {loss.item():.4f} | {ips:.2f} it/s")

        if iter_num % SAVE_INTERVAL == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                extra_name=f"iter_{iter_num}.pth",
            )
            log(f"중간 checkpoint 저장 완료: iter_{iter_num}.pth")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        extra_name=f"iter_{iter_num}_final.pth",
    )
    log("이번 실행 학습 종료 및 최종 저장 완료")

    try:
        sample = generate_text(model, sp, prompt="[대화 시작]\n화자1:", max_new_tokens=120)
        log("샘플 생성 결과:")
        log(sample)
    except Exception as e:
        log(f"샘플 생성 실패: {e}")

    log("=" * 60)


if __name__ == "__main__":
    main()