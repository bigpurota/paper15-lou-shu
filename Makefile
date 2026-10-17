PY      := .venv/bin/python
PYI     := $(PY) -I
CONFIG  := config.yaml
DAILY   := data/cache/daily.parquet
ZIP     := ../paper15-lou-shu-team12.zip

.PHONY: verify tests-log venv data tests sims run report figures p0 base upgrades upgrades-light ledger docs slides all all-light zip clean repro repro-final check-outputs
.DEFAULT_GOAL := all
.NOTPARALLEL:

STAMP   := .venv/.requirements-installed

venv: $(STAMP)
$(STAMP): requirements.txt
	@test -x $(PY) || { if command -v uv >/dev/null 2>&1; then uv venv .venv --python 3.12; \
	  else python3.12 -m venv .venv; fi; }
	@if command -v uv >/dev/null 2>&1; then uv pip install --python $(PY) -r requirements.txt; \
	  else $(PY) -m pip install -r requirements.txt; fi
	@touch $(STAMP)

data: $(DAILY)
$(DAILY): | $(STAMP)
	$(PYI) src/data_binance.py $(CONFIG)

tests: | $(STAMP)
	$(PYI) -m pytest -q --ignore=tests/test_manifests.py

sims: | $(STAMP)
	$(PYI) src/simulate.py $(CONFIG) --out results/simulations

run: $(DAILY) | $(STAMP)
	$(PYI) src/run.py $(CONFIG) --daily $(DAILY)

report: | base
	$(PYI) src/report.py

figures: | base
	$(PYI) src/figures.py

zip:
	@rm -f $(ZIP)
	@{ git ls-files --cached --others --exclude-standard; find results -type f 2>/dev/null; ls slides/*.pdf 2>/dev/null; } \
	  | grep -v -e '^data/' -e '^\.venv/' | sort -u \
	  | while read -r f; do [ -f "$$f" ] && echo "$$f"; done \
	  | zip -q -@ $(ZIP)
	@echo "wrote $(ZIP)"; unzip -l $(ZIP) | tail -1

repro: | $(STAMP)
	@test -n "$(CLONE)" || { echo "usage: make repro CLONE=<clone dir>"; exit 2; }
	$(PYI) src/repro_check.py $(CLONE) --log $(CLONE)/fresh_run.log

repro-final: | $(STAMP)
	@test -n "$(CLONE)" || { echo "usage: make repro-final CLONE=<clone dir>"; exit 2; }
	$(PYI) src/repro_check.py $(CLONE) --final --log $(CLONE)/final_run.log --make-log $(CLONE)/make_all.log

clean:
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__ $(ZIP)
	rm -f data/cache/panel_*.parquet

UPGRADES :=
-include mk/*.mk

$(UPGRADES): | base
base: | $(STAMP)
	@test -f results/LATEST && test -f data/cache/panel_$$(cat results/LATEST).parquet || $(MAKE) run

ledger: | $(STAMP)
	$(PYI) src/robustness_extra.py $(CONFIG) --ledger-only

docs: | $(STAMP)
	$(PYI) src/build_docs.py

DECK    := slides/paper15_team12_LouShu2017.pdf
slides: docs
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build deck.tex
	@! grep -q "Missing character" slides/build/deck.log || { grep "Missing character" slides/build/deck.log; exit 1; }
	cp slides/build/deck.pdf $(DECK)

p0: data tests sims run report figures
UPGRADES_ORDERED := $(filter upgrade-edge_spread,$(UPGRADES)) $(filter-out upgrade-edge_spread,$(UPGRADES))
upgrades: $(UPGRADES_ORDERED)
upgrades-light: $(filter-out upgrade-hf_lambda,$(UPGRADES_ORDERED))
all: p0 upgrades ledger docs check-outputs
all-light: p0 upgrades-light ledger docs check-outputs

check-outputs: | $(STAMP)
	$(PYI) -m pytest -q tests/test_manifests.py

verify: | $(STAMP)
	$(PYI) src/verify_manifests.py

tests-log: | $(STAMP)
	@mkdir -p results/tests_log; f=results/tests_log/pytest_$$(date -u +%Y%m%dT%H%M%SZ).txt; \
	{ echo "git HEAD: $$(git rev-parse HEAD 2>/dev/null || echo n/a)"; \
	  echo "dirty files: $$(git status --porcelain -- src tests config.yaml Makefile mk requirements.txt 2>/dev/null | wc -l | tr -d ' ')"; \
	  echo "code state: $$($(PYI) src/build_docs.py --code-hash)"; \
	  echo "date: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo; \
	  $(PYI) -m pytest -q 2>&1; } > $$f; tail -1 $$f; echo "wrote $$f"
