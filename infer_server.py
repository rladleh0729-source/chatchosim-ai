from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI
from pydantic import BaseModel


# =========================================================
# 기본 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TOKENIZER_DIR = BASE_DIR / "tokenizer"
ACTIVE_MODEL_FILE = BASE_DIR / "active_model.txt"

SPM_MODEL_PATH = TOKENIZER_DIR / "spm_korean_chat.model"
SPM_VOCAB_PATH = TOKENIZER_DIR / "spm_korean_chat.vocab"

LATEST_CKPT_PATH = CHECKPOINT_DIR / "latest_checkpoint.pth"
BEST_CKPT_PATH = CHECKPOINT_DIR / "best_checkpoint.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 요청 모델
# =========================================================
class GenerateRequest(BaseModel):
    message: str
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.1


class ActivateRequest(BaseModel):
    checkpoint: str


# =========================================================
# 모델 구조
# train.py 와 동일
# =========================================================
@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


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

    def forward(self, idx, targets=None):
        _, T = idx.shape
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


# =========================================================
# 전역 상태
# =========================================================
app = FastAPI()

sp: Optional[spm.SentencePieceProcessor] = None
model: Optional[GPTLanguageModel] = None
loaded_checkpoint_name: Optional[str] = None
loaded_checkpoint_path: Optional[str] = None
loaded_config: Optional[ModelConfig] = None
startup_warning: Optional[str] = None


# =========================================================
# 유틸
# =========================================================
def ensure_dirs() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def read_active_checkpoint_name() -> str:
    if ACTIVE_MODEL_FILE.exists():
        text = ACTIVE_MODEL_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text
    return "best"


def save_active_checkpoint_name(name: str) -> None:
    ACTIVE_MODEL_FILE.write_text(name, encoding="utf-8")


def resolve_checkpoint_path(name: str) -> Path:
    name = name.strip()

    if not name:
        raise FileNotFoundError("체크포인트 이름이 비어 있다.")

    if name.lower() == "latest":
        if not LATEST_CKPT_PATH.exists():
            raise FileNotFoundError(f"latest checkpoint 없음: {LATEST_CKPT_PATH}")
        return LATEST_CKPT_PATH

    if name.lower() == "best":
        if not BEST_CKPT_PATH.exists():
            raise FileNotFoundError(f"best checkpoint 없음: {BEST_CKPT_PATH}")
        return BEST_CKPT_PATH

    as_path = Path(name)
    if as_path.is_absolute():
        if not as_path.exists():
            raise FileNotFoundError(f"체크포인트 없음: {as_path}")
        return as_path

    candidate = CHECKPOINT_DIR / name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"체크포인트를 찾을 수 없다: {name}")


def load_sentencepiece() -> spm.SentencePieceProcessor:
    if not SPM_MODEL_PATH.exists():
        raise FileNotFoundError(f"SentencePiece 모델 없음: {SPM_MODEL_PATH}")

    processor = spm.SentencePieceProcessor()
    ok = processor.load(str(SPM_MODEL_PATH))
    if not ok:
        raise RuntimeError("SentencePiece 모델 로드 실패")
    return processor


def build_model_from_ckpt_config(ckpt_config: Dict[str, Any]) -> GPTLanguageModel:
    config = ModelConfig(
        vocab_size=int(ckpt_config["vocab_size"]),
        block_size=int(ckpt_config["block_size"]),
        n_embd=int(ckpt_config["n_embd"]),
        n_head=int(ckpt_config["n_head"]),
        n_layer=int(ckpt_config["n_layer"]),
        dropout=float(ckpt_config["dropout"]),
    )
    model_local = GPTLanguageModel(config).to(DEVICE)
    return model_local


def list_checkpoints() -> List[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    files = [p.name for p in CHECKPOINT_DIR.glob("*.pth") if p.is_file()]
    files.sort()
    return files


def clear_loaded_model() -> None:
    global sp, model, loaded_checkpoint_name, loaded_checkpoint_path, loaded_config
    sp = None
    model = None
    loaded_checkpoint_name = None
    loaded_checkpoint_path = None
    loaded_config = None


def load_checkpoint(checkpoint_name: str) -> None:
    global sp, model, loaded_checkpoint_name, loaded_checkpoint_path, loaded_config

    ckpt_path = resolve_checkpoint_path(checkpoint_name)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    ckpt_config = ckpt.get("config")
    if ckpt_config is None:
        raise RuntimeError("체크포인트에 config 정보가 없다.")

    processor = load_sentencepiece()
    vocab_size_from_sp = processor.vocab_size()
    vocab_size_from_ckpt = int(ckpt_config["vocab_size"])

    if vocab_size_from_sp != vocab_size_from_ckpt:
        raise RuntimeError(
            f"토크나이저 vocab({vocab_size_from_sp}) 와 체크포인트 vocab({vocab_size_from_ckpt}) 가 다르다."
        )

    model_local = build_model_from_ckpt_config(ckpt_config)
    model_local.load_state_dict(ckpt["model_state_dict"])
    model_local.eval()

    sp = processor
    model = model_local
    loaded_checkpoint_name = ckpt_path.name
    loaded_checkpoint_path = str(ckpt_path)
    loaded_config = model_local.config


@torch.no_grad()
def generate_text(
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> str:
    if sp is None or model is None or loaded_config is None:
        raise RuntimeError("아직 로드된 모델이 없다.")

    prompt = prompt.strip()
    if not prompt:
        return "메시지가 비어 있다."

    input_text = f"[대화 시작]\n화자1: {prompt}\n화자2:"
    ids = sp.encode(input_text, out_type=int)

    if not ids:
        bos_id = sp.bos_id()
        ids = [bos_id if bos_id >= 0 else 1]

    idx = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    eos_id = sp.eos_id()
    block_size = loaded_config.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if repetition_penalty != 1.0 and idx.numel() > 0:
            used_tokens = set(idx[0].tolist())
            for token_id in used_tokens:
                logits[0, token_id] /= repetition_penalty

        if temperature <= 0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)

            if 0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                sorted_probs[sorted_mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                sampled = torch.multinomial(sorted_probs, num_samples=1)
                idx_next = torch.gather(sorted_indices, -1, sampled)
            else:
                idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

        if eos_id >= 0 and int(idx_next.item()) == eos_id:
            break

    decoded = sp.decode(idx[0].tolist())

    reply = decoded
    if "화자2:" in decoded:
        reply = decoded.split("화자2:", 1)[1]

    if "화자1:" in reply:
        reply = reply.split("화자1:", 1)[0]

    reply = reply.strip()

    if not reply:
        reply = "응답 생성에 실패했다."

    return reply



@app.on_event("startup")
def startup_event() -> None:
    global startup_warning

    ensure_dirs()
    clear_loaded_model()
    startup_warning = None

    active_name = read_active_checkpoint_name()

    try:
        load_checkpoint(active_name)
    except Exception as e:
        startup_warning = str(e)
        print(f"[경고] 시작 시 모델 로드 실패: {startup_warning}")


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "UP",
        "device": DEVICE,
        "model_ready": model is not None,
        "loaded_checkpoint_name": loaded_checkpoint_name,
        "loaded_checkpoint_path": loaded_checkpoint_path,
        "checkpoint_count": len(list_checkpoints()),
        "tokenizer_model_exists": SPM_MODEL_PATH.exists(),
        "startup_warning": startup_warning,
    }


@app.get("/checkpoints")
def checkpoints():
    return {
        "success": True,
        "items": list_checkpoints(),
        "active": loaded_checkpoint_name,
    }


@app.post("/activate-model")
def activate_model(req: ActivateRequest):
    checkpoint = req.checkpoint.strip()

    if not checkpoint:
        return {
            "success": False,
            "message": "checkpoint 값이 비어 있다."
        }

    try:
        load_checkpoint(checkpoint)
        save_active_checkpoint_name(checkpoint)

        return {
            "success": True,
            "message": "활성 체크포인트 변경 완료",
            "active_checkpoint_name": loaded_checkpoint_name,
            "active_checkpoint_path": loaded_checkpoint_path,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"활성 체크포인트 변경 실패: {e}"
        }


@app.post("/generate")
def generate(req: GenerateRequest):
    text = req.message.strip()

    if not text:
        return {
            "success": False,
            "reply": "메시지가 비어 있다."
        }

    if model is None or sp is None or loaded_config is None:
        return {
            "success": False,
            "reply": "아직 학습된 체크포인트가 로드되지 않았다. 먼저 train.py로 학습하거나 체크포인트를 활성화해야 한다."
        }

    try:
        reply = generate_text(
            prompt=text,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )

        return {
            "success": True,
            "reply": reply,
            "checkpoint": loaded_checkpoint_name,
        }
    except Exception as e:
        return {
            "success": False,
            "reply": f"생성 실패: {e}"
        }
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "5001"))
    uvicorn.run("infer_server:app", host="0.0.0.0", port=port)