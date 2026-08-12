"""
Precompute frozen ProstT5 token embeddings for the multitask token cache.

The training architecture keeps ProstT5 frozen, so the expensive transformer
forward pass can be cached per unique chain sequence. The trainable adapter,
residue pooling, group pooling, pair MLP, and task heads still run during
training.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import T5EncoderModel

from config import (
  BACKBONE_EMBEDDING_CACHE_PATH,
  EMBEDDING_CACHE_MAX_TOKENS_PER_BATCH,
  MODEL_NAME,
  TRAIN_CACHE_PATH,
)
from model import token_ids_key


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"


def _iter_chain_token_tensors(payload):
  for split_payload in payload["splits"].values():
    for group_key in ("group1_input_ids", "group2_input_ids"):
      for sample_chains in split_payload[group_key]:
        yield from sample_chains


def _unique_chain_token_tensors(payload):
  unique = {}
  for input_ids in _iter_chain_token_tensors(payload):
    unique.setdefault(token_ids_key(input_ids), input_ids)
  return unique


def _pack_token_batches(items):
  batches = []
  current = []
  current_tokens = 0

  for key, input_ids in sorted(items, key=lambda item: len(item[1])):
    sequence_tokens = len(input_ids)
    would_exceed = current and current_tokens + sequence_tokens > EMBEDDING_CACHE_MAX_TOKENS_PER_BATCH
    if current and would_exceed:
      batches.append(current)
      current = []
      current_tokens = 0

    current.append((key, input_ids))
    current_tokens += sequence_tokens

  if current:
    batches.append(current)

  return batches


def main():
  if not TRAIN_CACHE_PATH.exists():
    raise FileNotFoundError(f"Missing tokenized cache at {TRAIN_CACHE_PATH}. Run tokenize_data.py first.")

  print(f"Loading tokenized cache from {TRAIN_CACHE_PATH}")
  payload = torch.load(TRAIN_CACHE_PATH, map_location="cpu")
  pad_token_id = payload["config"]["pad_token_id"]
  unique_tokens = _unique_chain_token_tensors(payload)
  batches = _pack_token_batches(list(unique_tokens.items()))

  print(f"Unique chain sequences={len(unique_tokens)} batches={len(batches)}")
  model = T5EncoderModel.from_pretrained(MODEL_NAME).to(DEVICE)
  if DEVICE.type == "cuda":
    model.bfloat16()
  model.eval()

  embeddings = {}
  with torch.no_grad():
    for batch in tqdm(batches, desc="Cache ProstT5 embeddings"):
      keys = [key for key, _ in batch]
      input_ids = [ids for _, ids in batch]
      padded_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
      attention_mask = padded_ids.ne(pad_token_id).long()

      padded_ids = padded_ids.to(DEVICE, non_blocking=PIN_MEMORY)
      attention_mask = attention_mask.to(DEVICE, non_blocking=PIN_MEMORY)

      with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=AMP_ENABLED):
        outputs = model(input_ids=padded_ids.long(), attention_mask=attention_mask)

      hidden = outputs.last_hidden_state.detach().cpu().to(torch.bfloat16)
      for key, ids, token_embeddings in zip(keys, input_ids, hidden):
        embeddings[key] = token_embeddings[: len(ids)].contiguous()

  BACKBONE_EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    {
      "cache_format": "prostt5_backbone_token_embeddings_v1",
      "model_name": MODEL_NAME,
      "tokenized_cache_path": str(TRAIN_CACHE_PATH),
      "embedding_dtype": "bfloat16",
      "num_sequences": len(embeddings),
      "embeddings": embeddings,
    },
    BACKBONE_EMBEDDING_CACHE_PATH,
  )
  print(f"Saved frozen backbone embeddings -> {BACKBONE_EMBEDDING_CACHE_PATH}")


if __name__ == "__main__":
  main()
