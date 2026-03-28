PYTHON ?= python3
DATA_DIR ?= data/raw
PROCESSED_DIR ?= data/processed

.PHONY: all data phonetics tokenizer pretrain finetune eval test verify dashboard export-dashboard clean

all: data phonetics tokenizer pretrain finetune eval

# Download and clean training data
data: data-download data-clean

data-download:
	$(PYTHON) data/download.py --output-dir $(DATA_DIR)

data-clean:
	$(PYTHON) data/clean.py --input-dir $(DATA_DIR) --output-dir $(PROCESSED_DIR)

# Build phonetic confusion databases from espeak-ng
phonetics: phonetics/ipa_db.json phonetics/confusion_db.json phonetics/homophone_sets.json

phonetics/ipa_db.json:
	$(PYTHON) phonetics/build_ipa_db.py --output phonetics/ipa_db.json

phonetics/confusion_db.json: phonetics/ipa_db.json
	$(PYTHON) phonetics/build_confusion_db.py \
		--ipa-db phonetics/ipa_db.json \
		--output phonetics/confusion_db.json

phonetics/homophone_sets.json: phonetics/ipa_db.json
	$(PYTHON) phonetics/homophones.py \
		--ipa-db phonetics/ipa_db.json \
		--output phonetics/homophone_sets.json

# Train SentencePiece BPE tokenizer on corpus
tokenizer: tokenizer/tokenizer.model

tokenizer/tokenizer.model: $(PROCESSED_DIR)
	$(PYTHON) tokenizer/train_tokenizer.py \
		--input-dir $(PROCESSED_DIR) \
		--output tokenizer/tokenizer.model \
		--vocab-size 32000

# Verify model architecture (~385M params)
verify:
	PYTHONPATH=. $(PYTHON) model/verify_params.py

# Phase 1: Denoising pretraining (~5-7 days, RTX 3090)
pretrain:
	PYTHONPATH=. $(PYTHON) training/pretrain.py \
		--data-dir $(PROCESSED_DIR) \
		--tokenizer tokenizer/tokenizer.model \
		--checkpoint-dir checkpoints/pretrain

# Phase 2: Correction fine-tuning (~2-3 days)
finetune:
	PYTHONPATH=. $(PYTHON) training/finetune.py \
		--data-dir $(PROCESSED_DIR) \
		--tokenizer tokenizer/tokenizer.model \
		--pretrained checkpoints/pretrain/best.pt \
		--checkpoint-dir checkpoints/finetune

# Evaluation
eval:
	PYTHONPATH=. $(PYTHON) eval/benchmarks.py \
		--model checkpoints/finetune/best.pt \
		--tokenizer tokenizer/tokenizer.model \
		--verbose

# Export static dashboard snapshot to docs/
export-dashboard:
	PYTHONPATH=. $(PYTHON) tools/export_dashboard.py

# Web dashboard for monitoring training
dashboard:
	PYTHONPATH=. $(PYTHON) tools/web_dashboard.py --host 0.0.0.0

# Run corruption engine tests
test:
	$(PYTHON) -m pytest corruption/test_corruption.py -v

# Remove built artifacts
clean:
	rm -rf phonetics/ipa_db.json phonetics/confusion_db.json phonetics/homophone_sets.json
	rm -rf tokenizer/tokenizer.model tokenizer/tokenizer.vocab
	rm -rf $(PROCESSED_DIR)
