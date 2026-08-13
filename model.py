"""
Shared model and data utilities for multitask protein-group pair training.
"""

import math
import hashlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

from config import (
  ADAPTER_DIM,
  CLASSIFICATION_HEAD_HIDDEN,
  DROPOUT,
  GROUP_POOL_HIDDEN,
  PAIR_MLP_HIDDEN,
  RESIDUE_POOL_HIDDEN,
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


class Adapter(nn.Module):
  def __init__(self, input_dim, adapter_dim=ADAPTER_DIM, dropout_prob=DROPOUT):
    super().__init__()
    self.norm = nn.LayerNorm(input_dim)
    self.down_project = nn.Linear(input_dim, adapter_dim)
    self.activation = nn.GELU()
    self.up_project = nn.Linear(adapter_dim, input_dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.scale = nn.Parameter(torch.tensor(1e-3))
    nn.init.normal_(self.down_project.weight, std=1e-3)
    nn.init.normal_(self.up_project.weight, std=1e-3)
    nn.init.zeros_(self.up_project.bias)

  def forward(self, x):
    x_norm = self.norm(x)
    return self.scale * self.dropout(self.up_project(self.activation(self.down_project(x_norm))))


class AttentionPool(nn.Module):
  def __init__(self, d_model, hidden, dropout=DROPOUT):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(d_model, hidden),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.context = nn.Linear(hidden, 1, bias=False)

  def forward(self, x, mask):
    h = self.proj(x)
    scores = self.context(h).squeeze(-1)
    scores = scores.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(scores, dim=1)
    return torch.bmm(attn.unsqueeze(1), x).squeeze(1)


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
    adapter_dim=ADAPTER_DIM,
    dropout=DROPOUT,
    classification_head_hidden=CLASSIFICATION_HEAD_HIDDEN,
  ):
    super().__init__()
    self.base = base_model
    if self.base is not None:
      for param in self.base.parameters():
        param.requires_grad = False

    self.adapter = Adapter(embed_dim, adapter_dim, dropout_prob=dropout)
    self.residue_pool = AttentionPool(embed_dim, hidden=RESIDUE_POOL_HIDDEN, dropout=dropout)
    self.group_pool = AttentionPool(embed_dim, hidden=GROUP_POOL_HIDDEN, dropout=dropout)
    self.pair_mlp = nn.Sequential(
      nn.LayerNorm(embed_dim * 3),
      nn.Linear(embed_dim * 3, PAIR_MLP_HIDDEN),
      nn.GELU(),
      nn.Dropout(dropout),
    )
    self.heads = nn.ModuleDict()

    for task_name in task_order:
      self.heads[task_name] = PairTaskHead(PAIR_MLP_HIDDEN, task_output_dims[task_name], classification_head_hidden, dropout=dropout)

  def encode_shared_tokens(self, input_ids, attention_mask, precomputed_embeddings=None):
    if precomputed_embeddings is None:
      if self.base is None:
        raise ValueError("A base model is required when precomputed embeddings are not provided.")
      out = self.base(input_ids=input_ids.long(), attention_mask=attention_mask.long())
      token_repr = out.last_hidden_state
    else:
      token_repr = precomputed_embeddings.to(dtype=self.adapter.norm.weight.dtype)
    return token_repr + self.adapter(token_repr)

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
