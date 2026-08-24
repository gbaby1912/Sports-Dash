# Sports-Dash — ATP/WTA match prediction
#
# Quick start with the real public archives:
#     make install && make data && make train && make serve
#
# Quick start with no network access (synthetic data, full pipeline):
#     make install && make demo && make serve

PYTHON ?= python3
TOURS  ?= atp wta
START  ?= 2000

.PHONY: help install data synth ingest features train backtest serve test predict rankings clean distclean demo all

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime and development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

data: ingest ## Download the public archives and build the match table
fetch: ## Download the Sackmann ATP/WTA archives into data/raw
	$(PYTHON) -m tennisdash.cli fetch --tours $(TOURS) --start-year $(START)

ingest: fetch ## Normalise the raw archives into data/processed
	$(PYTHON) -m tennisdash.cli ingest --tours $(TOURS)

synth: ## Generate an offline synthetic dataset (no network needed)
	$(PYTHON) -m tennisdash.cli synth --tours $(TOURS)

features: ## Build the model feature matrix
	$(PYTHON) -m tennisdash.cli features

train: ## Train the model, run the backtest, write the bundle
	$(PYTHON) -m tennisdash.cli train --rebuild

backtest: ## Walk-forward evaluation, printed as a table
	$(PYTHON) -m tennisdash.cli backtest

serve: ## Run the dashboard at http://127.0.0.1:8000
	$(PYTHON) -m tennisdash.cli serve

test: ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

predict: ## Example: make predict P1="Alcaraz" P2="Sinner" SURFACE=Clay
	$(PYTHON) -m tennisdash.cli predict "$(P1)" "$(P2)" --surface $(or $(SURFACE),Hard) \
		--tour $(or $(TOUR),atp) --best-of $(or $(BO),3)

rankings: ## Elo leaderboard
	$(PYTHON) -m tennisdash.cli rankings --tour $(or $(TOUR),atp) --surface $(or $(SURFACE),)

demo: synth ingest train ## Full offline pipeline: synthetic data through to a trained model

all: data train ## Full pipeline against the real archives

clean: ## Remove derived data and model artifacts (keeps the raw cache)
	rm -rf data/processed/* data/artifacts/*
	@touch data/processed/.gitkeep data/artifacts/.gitkeep

distclean: clean ## Also remove the raw download cache
	rm -rf data/raw/*
	@touch data/raw/.gitkeep
