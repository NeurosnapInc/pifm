from pathlib import Path

### Backbone
MODEL_NAME = "Rostlab/ProstT5"
GLOBAL_SEED = 1

### Aggregation
AGGREGATED_DB_PATH = Path("data/aggregated/aggregated.duckdb")
INTACT_ARCHIVE_PATH = Path("data/raw/intact_all_2026_07_03.zip")
INTACT_SPECIES_TAXID = "9606"
INTACT_INTERACTOR_TYPES = ("protein", "peptide")
INTACT_INTERACTION_TYPES = ("physical association", "direct interaction")

### Split Settings
SPLIT_SEED = GLOBAL_SEED
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
SPLIT_STRATEGY = "cluster"  # "random", "protein", or "cluster"
CLUSTER_MIN_SEQ_ID = 0.5
CLUSTER_COVERAGE = 0.8
MMSEQS_BINARY = "mmseqs"
SEQUENCE_CLUSTER_FASTA_PATH = Path("data/tokenized/split_sequences.fasta")
SEQUENCE_CLUSTER_TSV_PATH = Path("data/tokenized/split_sequence_clusters.tsv")
SEQUENCE_CLUSTER_WORK_DIR = Path("data/tokenized/mmseqs_tmp")

### Tokenization
MAX_LENGTH = 1024 * 2
TOKENIZED_DATA_DIR = Path("data/tokenized")
TRAIN_CACHE_PATH = TOKENIZED_DATA_DIR / "multitask_group_pair_prostt5_tokens.pt"

### Optimization
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 10
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.05
PATIENCE = 3
TRAINING_SEED = GLOBAL_SEED
BATCH_SAMPLER_SEED = GLOBAL_SEED

### Early Stopping / Checkpoint Selection
MIN_CLASSIFICATION_VAL_LABELS = 100
MIN_REGRESSION_VAL_LABELS = 100
CLASSIFICATION_SELECTION_METRIC = "auroc"  # "auroc" or "balanced_accuracy"
REGRESSION_SELECTION_METRIC = "normalized_mae"  # "normalized_mae" or "pearson"

### Affinity Regression
REGRESSION_LOSS = "huber"  # "mse" or "huber"
REGRESSION_HUBER_DELTA = 1.0
AFFINITY_NORMALIZATION = "source"  # "global" or "source"
MIN_SOURCE_AFFINITY_LABELS = 20

### Architecture
ADAPTER_DIM = 64
DROPOUT = 0.1
RESIDUE_POOL_HIDDEN = 256
GROUP_POOL_HIDDEN = 256
PAIR_MLP_HIDDEN = 512
CLASSIFICATION_HEAD_HIDDEN = 256
REGRESSION_HEAD_HIDDEN = 512

### Token-Capped Batching
TRAIN_MAX_TOKENS_PER_BATCH = 49152
EVAL_MAX_TOKENS_PER_BATCH = 65536