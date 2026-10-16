# Lou & Shu (2017) on Binance spot. `make all` is the single command (data -> tests -> sims -> run -> report ->
# upgrades -> ledger -> docs). It has been run on a clean clone (with cached downloads), and a from-scratch clone
# redid the downloads on an older commit: results/repro/ and README "Reproducibility".
PY      := .venv/bin/python
PYI     := $(PY) -I
CONFIG  := config.yaml
DAILY   := data/cache/daily.parquet
ZIP     := ../paper15-lou-shu-team12.zip

.PHONY: tests-log venv data tests sims run report figures p0 base upgrades upgrades-light ledger docs slides all all-light zip clean repro repro-final check-outputs
.DEFAULT_GOAL := all
# stages depend on each other through files in results/ (LATEST, upgrade CSVs): keep them sequential
.NOTPARALLEL:

STAMP   := .venv/.requirements-installed

venv: $(STAMP)
# uses uv when it is installed, otherwise the standard library venv + pip of python3.12
$(STAMP): requirements.txt
	@test -x $(PY) || { if command -v uv >/dev/null 2>&1; then uv venv .venv --python 3.12; \
	  else python3.12 -m venv .venv; fi; }
	@if command -v uv >/dev/null 2>&1; then uv pip install --python $(PY) -r requirements.txt; \
	  else $(PY) -m pip install -r requirements.txt; fi
	@touch $(STAMP)

# download is skipped when the cached daily panel exists (delete it to re-download)
data: $(DAILY)
$(DAILY): | $(STAMP)
	$(PYI) src/data_binance.py $(CONFIG)

# code tests; the output contract (tests/test_manifests.py) checks files a run writes, so it runs in
# check-outputs at the END of make all (here it would only see the previous run's outputs)
tests: | $(STAMP)
	$(PYI) -m pytest -q --ignore=tests/test_manifests.py

sims: | $(STAMP)
	$(PYI) src/simulate.py $(CONFIG) --out results/simulations

# each run writes a new results/<run_id>/ and points results/LATEST to it
run: $(DAILY) | $(STAMP)
	$(PYI) src/run.py $(CONFIG) --daily $(DAILY)

report: | base
	$(PYI) src/report.py

figures: | base
	$(PYI) src/figures.py

# tracked + untracked-but-not-ignored files + results/ + slides/*.pdf; never data/ or .venv
zip:
	@rm -f $(ZIP)
	@{ git ls-files --cached --others --exclude-standard; find results -type f 2>/dev/null; ls slides/*.pdf 2>/dev/null; } \
	  | grep -v -e '^data/' -e '^\.venv/' | sort -u \
	  | while read -r f; do [ -f "$$f" ] && echo "$$f"; done \
	  | zip -q -@ $(ZIP)
	@echo "wrote $(ZIP)"; unzip -l $(ZIP) | tail -1

# reproducibility evidence from a clean clone in CLONE=<dir> (src/repro_check.py; README "Reproducibility"):
#   repro       from-scratch clone (empty data/): reads $(CLONE)/fresh_run.log and $(CLONE)/make_*.log
#   repro-final clone that ran `make all`: reads $(CLONE)/final_run.log and $(CLONE)/make_all.log
repro: | $(STAMP)
	@test -n "$(CLONE)" || { echo "usage: make repro CLONE=<clone dir>"; exit 2; }
	$(PYI) src/repro_check.py $(CLONE) --log $(CLONE)/fresh_run.log

repro-final: | $(STAMP)
	@test -n "$(CLONE)" || { echo "usage: make repro-final CLONE=<clone dir>"; exit 2; }
	$(PYI) src/repro_check.py $(CLONE) --final --log $(CLONE)/final_run.log --make-log $(CLONE)/make_all.log

clean:
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__ $(ZIP)
	rm -f data/cache/panel_*.parquet

# upgrade modules register themselves in mk/*.mk via `UPGRADES += upgrade-<name>`
# (included last so that PY/PYI/CONFIG/STAMP are already defined)
UPGRADES :=
-include mk/*.mk

# upgrades and the report read the base run that results/LATEST points to, including its panel cache
# (data/cache/panel_<run_id>.parquet, not in git): on a fresh clone or after `make clean` make a new run first
$(UPGRADES): | base
base: | $(STAMP)
	@test -f results/LATEST && test -f data/cache/panel_$$(cat results/LATEST).parquet || $(MAKE) run

# the multiple-testing ledger reads every upgrade's CSVs, so it is rebuilt after all of them
ledger: | $(STAMP)
	$(PYI) src/robustness_extra.py $(CONFIG) --ledger-only

# README.md and slides/PLAN.md: every number is refilled from results/ (see src/build_docs.py); the same
# values go to slides/gen/values.tex and the slide tables to slides/gen/*.tex
docs: | $(STAMP)
	$(PYI) src/build_docs.py

# slide deck: slides/deck.tex (text only, every number is a \V{key} macro from slides/gen/values.tex)
# -> slides/build/deck.pdf -> $(DECK). Needs xelatex + latexmk (TeX Live) and the Arial font.
DECK    := slides/paper15_team12_LouShu2017.pdf
slides: docs
	cd slides && latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build deck.tex
	@! grep -q "Missing character" slides/build/deck.log || { grep "Missing character" slides/build/deck.log; exit 1; }
	cp slides/build/deck.pdf $(DECK)

p0: data tests sims run report figures
# hf_lambda reads data/cache/edge_monthly.parquet written by edge_spread, so edge_spread runs first
# (prerequisites run left to right under .NOTPARALLEL, whatever order mk/*.mk were included in)
UPGRADES_ORDERED := $(filter upgrade-edge_spread,$(UPGRADES)) $(filter-out upgrade-edge_spread,$(UPGRADES))
upgrades: $(UPGRADES_ORDERED)
# without hf_lambda (it downloads ~5 GB on the first run): its committed results/upgrades/hf_lambda/ stay as they are
upgrades-light: $(filter-out upgrade-hf_lambda,$(UPGRADES_ORDERED))
all: p0 upgrades ledger docs check-outputs
all-light: p0 upgrades-light ledger docs check-outputs

# run LAST: the manifest contract (no absolute paths, required keys) on the outputs this run just wrote;
# `make tests` inside p0 runs before the upgrades and therefore sees the previous outputs
check-outputs: | $(STAMP)
	$(PYI) -m pytest -q tests/test_manifests.py

# saved pytest log (all tests incl. the output contract) with the code state it ran on; read by build_docs
tests-log: | $(STAMP)
	@mkdir -p results/tests_log; f=results/tests_log/pytest_$$(date -u +%Y%m%dT%H%M%SZ).txt; \
	{ echo "git HEAD: $$(git rev-parse HEAD)"; \
	  echo "dirty files: $$(git status --porcelain -- src tests config.yaml Makefile mk requirements.txt | wc -l | tr -d ' ')"; \
	  echo "code state: $$($(PYI) src/build_docs.py --code-hash)"; \
	  echo "date: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo; \
	  $(PYI) -m pytest -q 2>&1; } > $$f; tail -1 $$f; echo "wrote $$f"
