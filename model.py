"""
Shared model and data utilities for multitask protein-group pair training.
"""

import math
import hashlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

from config import (
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  PAIR_MLP_HIDDEN,
)


def token_ids_key(input_ids: torch.Tensor) -> str:
  ids = input_ids.detach().cpu().to(torch.int32).contiguous()
  return hashlib.sha256(ids.numpy().tobytes()).hexdigest()


def load_backbone_embedding_cache(path, expected_model_name=None, expected_tokenized_cache_path=None):
  payload = torch.load(path, map_location="cpu")
  if expected_model_name is not None and payload.get("model_name") != expected_model_name:
    raise ValueError(
      f"Embedding cache model mismatch: expected {expected_model_name!r}, "
      f"found {payload.get('model_name')!r} in {path}"
    )
  if expected_tokenized_cache_path is not None and payload.get("tokenized_cache_path") != str(expected_tokenized_cache_path):
    raise ValueError(
      f"Embedding cache tokenized path mismatch: expected {str(expected_tokenized_cache_path)!r}, "
      f"found {payload.get('tokenized_cache_path')!r} in {path}. Re-run cache_embeddings.py."
    )
  return payload["embeddings"], payload


class MultiTaskGroupPairDataset(Dataset):
  def __init__(self, split_payload, embedding_cache=None):
    self.samples = []
    for idx, length in enumerate(split_payload["lengths"]):
      group1_input_ids = split_payload["group1_input_ids"][idx]
      group2_input_ids = split_payload["group2_input_ids"][idx]
      sample = {
        "group1_input_ids": group1_input_ids,
        "group2_input_ids": group2_input_ids,
        "raw_labels": split_payload["raw_labels"][idx],
        "normalized_labels": split_payload["normalized_labels"][idx],
        "label_mask": split_payload["label_mask"][idx],
        "length": int(length),
        "source": split_payload.get("sources", ["unknown"] * len(split_payload["lengths"]))[idx],
      }
      if embedding_cache is not None:
        sample["group1_embeddings"] = [embedding_cache[token_ids_key(ids)] for ids in group1_input_ids]
        sample["group2_embeddings"] = [embedding_cache[token_ids_key(ids)] for ids in group2_input_ids]
      self.samples.append(sample)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx]


class MultiTaskBatchSampler(Sampler):
  def __init__(self, dataset, batch_size, shuffle=False, seed=0, sample_weights=None, max_tokens_per_batch=None):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.sample_weights = sample_weights
    self.max_tokens_per_batch = max_tokens_per_batch
    self.epoch = 0
    self.pool_size = batch_size * 50
    self.num_samples = len(dataset)

  def _pack_batches(self, indices):
    if self.max_tokens_per_batch is None:
      return [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

    batches = []
    current_batch = []
    current_tokens = 0

    for idx in indices:
      sample_tokens = self.dataset.samples[idx]["length"]
      would_exceed_tokens = current_batch and current_tokens + sample_tokens > self.max_tokens_per_batch
      if current_batch and (would_exceed_tokens or len(current_batch) >= self.batch_size):
        batches.append(current_batch)
        current_batch = []
        current_tokens = 0

      current_batch.append(idx)
      current_tokens += sample_tokens

    if current_batch:
      batches.append(current_batch)

    return batches

  def __iter__(self):
    if not self.shuffle:
      indices = list(range(len(self.dataset)))
      indices.sort(key=lambda idx: self.dataset.samples[idx]["length"])
      return iter(self._pack_batches(indices))

    generator = torch.Generator()
    generator.manual_seed(self.seed + self.epoch)
    self.epoch += 1

    sampled_indices = torch.multinomial(
      self.sample_weights,
      self.num_samples,
      replacement=True,
      generator=generator,
    ).tolist()

    batches = []
    for start in range(0, len(sampled_indices), self.pool_size):
      pool = sampled_indices[start:start + self.pool_size]
      pool.sort(key=lambda idx: self.dataset.samples[idx]["length"])
      batches.extend(self._pack_batches(pool))

    order = torch.randperm(len(batches), generator=generator).tolist()
    return iter([batches[idx] for idx in order])

  def __len__(self):
    return math.ceil(self.num_samples / self.batch_size)


class LoRALinear(nn.Module):
  """Add a rank-8 update to a frozen attention projection.

  A is randomly initialized and B starts at zero, preserving the pretrained
  function initially. Updates use alpha/rank scaling and float32 trainable
  parameters even when the frozen backbone weights use bfloat16.
  """

  rank = 8
  alpha = 16
  dropout_prob = 0.1

  def __init__(self, projection):
    super().__init__()
    self.projection = projection
    self.projection.requires_grad_(False)
    self.lora_A = nn.Linear(projection.in_features, self.rank, bias=False, device=projection.weight.device)
    self.lora_B = nn.Linear(self.rank, projection.out_features, bias=False, device=projection.weight.device)
    nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
    nn.init.zeros_(self.lora_B.weight)
    self.dropout = nn.Dropout(self.dropout_prob)

  def forward(self, x):
    original = self.projection(x)
    update = self.lora_B(self.lora_A(self.dropout(x).to(self.lora_A.weight.dtype)))
    return original + (update * (self.alpha / self.rank)).to(original.dtype)


class MaxPool(nn.Module):
  """Take each feature's maximum over valid token embeddings.

  Residue pooling uses the tokenizer's attention mask, including special tokens;
  mask padding with negative infinity so it cannot dominate negative features.
  Fully masked inputs return zeros.
  """

  def forward(self, x, mask):
    valid = mask.bool().unsqueeze(-1)
    pooled = x.masked_fill(~valid, float("-inf")).max(dim=1).values
    return torch.where(valid.any(dim=1), pooled, torch.zeros_like(pooled))


class MeanPool(nn.Module):
  """Average valid chain embeddings with equal weight for each chain.

  Accumulate in float32 for mixed-precision stability. Group pooling receives
  one vector per chain, so longer chains do not receive additional weight.
  """

  def forward(self, x, mask):
    valid = mask.bool().unsqueeze(-1)
    total = x.float().masked_fill(~valid, 0.0).sum(dim=1)
    count = valid.sum(dim=1).clamp_min(1)
    return (total / count).to(dtype=x.dtype)


class PairTaskHead(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_dim, dropout=DROPOUT):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(input_dim),
      nn.Linear(input_dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, output_dim),
    )

  def forward(self, x):
    return self.net(x)


class MultiTaskGroupPairModel(nn.Module):
  def __init__(
    self,
    base_model,
    task_order,
    task_output_dims,
    embed_dim,
    task_metas=None,
    dropout=DROPOUT,
    classification_head_hidden=CLASSIFICATION_HEAD_HIDDEN,
  ):
    super().__init__()
    if base_model is None:
      raise ValueError("LoRA requires live ProstT5 token encoding; backbone embedding caches cannot be used.")
    self.base = base_model
    self.base.requires_grad_(False)
    attention_layers = [module for name, module in self.base.named_modules() if name.endswith(".SelfAttention")]
    if not attention_layers:
      raise ValueError("No T5 self-attention layers found for LoRA injection.")
    for attention in attention_layers:
      for name in ("q", "v"):
        projection = getattr(attention, name)
        if not isinstance(projection, nn.Linear):
          raise ValueError(f"Expected an unmodified linear {name} projection for LoRA injection.")
        setattr(attention, name, LoRALinear(projection))

    self.residue_pool = MaxPool()
    self.group_pool = MeanPool()
    self.pair_mlp = nn.Sequential(
      nn.LayerNorm(embed_dim * 3),
      nn.Linear(embed_dim * 3, PAIR_MLP_HIDDEN),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.heads = nn.ModuleDict()

    for task_name in task_order:
      self.heads[task_name] = PairTaskHead(PAIR_MLP_HIDDEN, task_output_dims[task_name], classification_head_hidden, dropout=dropout)

  def lora_state_dict(self):
    """Return only LoRA matrices, excluding the frozen pretrained weights."""
    return {
      f"{name}.{key}": value
      for name, module in self.base.named_modules() if isinstance(module, LoRALinear)
      for key, value in (("lora_A.weight", module.lora_A.weight), ("lora_B.weight", module.lora_B.weight))
    }

  def load_lora_state_dict(self, state):
    """Load all LoRA matrices strictly, rejecting missing or extra keys/shapes."""
    expected = self.lora_state_dict()
    if set(state) != set(expected):
      raise ValueError("LoRA checkpoint keys do not match the current backbone projections.")
    if any(state[key].shape != value.shape for key, value in expected.items()):
      raise ValueError("LoRA checkpoint matrix shapes do not match the current architecture.")
    with torch.no_grad():
      for key, value in expected.items():
        value.copy_(state[key])

  def encode_shared_tokens(self, input_ids, attention_mask, precomputed_embeddings=None):
    if precomputed_embeddings is not None:
      raise ValueError("Frozen backbone embeddings bypass LoRA; provide token IDs instead.")
    out = self.base(input_ids=input_ids.long(), attention_mask=attention_mask.long())
    return out.last_hidden_state.to(self.pair_mlp[0].weight.dtype)

  def _pool_group(self, chain_embeddings, chain_to_sample, chain_to_group, batch_size: int, group_id: int):
    group_embeddings = []
    for sample_idx in range(batch_size):
      mask = (chain_to_sample == sample_idx) & (chain_to_group == group_id)
      sample_chains = chain_embeddings[mask]
      if sample_chains.shape[0] == 1:
        group_embeddings.append(sample_chains[0])
        continue
      pooled = self.group_pool(
        sample_chains.unsqueeze(0),
        torch.ones((1, sample_chains.shape[0]), dtype=torch.long, device=sample_chains.device),
      ).squeeze(0)
      group_embeddings.append(pooled)
    return torch.stack(group_embeddings, dim=0)

  def _pair_features(self, group1_embeddings, group2_embeddings):
    return torch.cat(
      [
        group1_embeddings + group2_embeddings,
        torch.abs(group1_embeddings - group2_embeddings),
        group1_embeddings * group2_embeddings,
      ],
      dim=-1,
    )

  def forward(self, input_ids, attention_mask, chain_to_sample, chain_to_group, batch_size, precomputed_embeddings=None):
    shared_tokens = self.encode_shared_tokens(input_ids, attention_mask, precomputed_embeddings=precomputed_embeddings)
    chain_embeddings = self.residue_pool(shared_tokens, attention_mask)
    group1_embeddings = self._pool_group(chain_embeddings, chain_to_sample, chain_to_group, batch_size, group_id=0)
    group2_embeddings = self._pool_group(chain_embeddings, chain_to_sample, chain_to_group, batch_size, group_id=1)
    pair_hidden = self.pair_mlp(self._pair_features(group1_embeddings, group2_embeddings))
    return {
      task_name: head(pair_hidden)
      for task_name, head in self.heads.items()
    }


def unwrap_model(model):
  return model._orig_mod if hasattr(model, "_orig_mod") else model


def output_dim_from_meta(meta, labels, mask):
  if meta["dtype"] != "bool":
    raise ValueError(f"Unsupported downstream task dtype={meta['dtype']!r}; only classification is enabled.")
  observed = labels[mask]
  if observed.numel() == 0:
    raise ValueError(f"Task '{meta['task_name']}' has no observed labels in train split.")
  if meta["num_classes"] is not None:
    return int(meta["num_classes"])
  return int(observed.max().item()) + 1


def collate_multitask_batch(batch, pad_token_id, include_sources=False):
  flat_input_ids = []
  flat_embeddings = []
  chain_to_sample = []
  chain_to_group = []
  use_embeddings = "group1_embeddings" in batch[0]

  for sample_idx, sample in enumerate(batch):
    group1_values = sample["group1_embeddings"] if use_embeddings else sample["group1_input_ids"]
    group2_values = sample["group2_embeddings"] if use_embeddings else sample["group2_input_ids"]
    for value in group1_values:
      if use_embeddings:
        flat_embeddings.append(value)
      else:
        flat_input_ids.append(value)
      chain_to_sample.append(sample_idx)
      chain_to_group.append(0)
    for value in group2_values:
      if use_embeddings:
        flat_embeddings.append(value)
      else:
        flat_input_ids.append(value)
      chain_to_sample.append(sample_idx)
      chain_to_group.append(1)

  if use_embeddings:
    padded_ids = None
    padded_embeddings = nn.utils.rnn.pad_sequence(flat_embeddings, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([embedding.shape[0] for embedding in flat_embeddings], dtype=torch.long)
    positions = torch.arange(padded_embeddings.shape[1]).unsqueeze(0)
    attention_mask = positions.lt(lengths.unsqueeze(1)).long()
  else:
    padded_ids = nn.utils.rnn.pad_sequence(flat_input_ids, batch_first=True, padding_value=pad_token_id)
    padded_embeddings = None
    attention_mask = padded_ids.ne(pad_token_id).long()
  raw_labels = torch.stack([sample["raw_labels"] for sample in batch])
  normalized_labels = torch.stack([sample["normalized_labels"] for sample in batch])
  label_mask = torch.stack([sample["label_mask"] for sample in batch])
  output = (
    padded_ids,
    padded_embeddings,
    attention_mask,
    torch.tensor(chain_to_sample, dtype=torch.long),
    torch.tensor(chain_to_group, dtype=torch.long),
    raw_labels,
    normalized_labels,
    label_mask,
  )
  if include_sources:
    return output + ([sample["source"] for sample in batch],)
  return output
