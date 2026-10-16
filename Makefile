# Lou & Shu (2017) on Binance spot: one command reproduces everything (`make all`).
PY      := .venv/bin/python
PYI     := $(PY) -I
CONFIG  := config.yaml
DAILY   := data/cache/daily.parquet
ZIP     := ../paper15-lou-shu-team12.zip

.PHONY: venv data tests sims run report figures all zip clean

all: data tests sims run report figures

STAMP   := .venv/.requirements-installed

venv: $(STAMP)
$(STAMP): requirements.txt
	@test -x $(PY) || uv venv .venv --python 3.12
	uv pip install --python $(PY) -r requirements.txt
	@touch $(STAMP)

# download is skipped when the cached daily panel exists (delete it to re-download)
data: $(DAILY)
$(DAILY): | $(STAMP)
	$(PYI) src/data_binance.py $(CONFIG)

tests: | $(STAMP)
	$(PYI) -m pytest -q

sims: | $(STAMP)
	$(PYI) src/simulate.py $(CONFIG) --out results/simulations

# each run writes a new results/<run_id>/ and points results/LATEST to it
run: $(DAILY) | $(STAMP)
	$(PYI) src/run.py $(CONFIG) --daily $(DAILY)

report: | $(STAMP)
	$(PYI) src/report.py

figures: | $(STAMP)
	$(PYI) src/figures.py

# tracked + untracked-but-not-ignored files + results/ + slides/*.pdf; never data/ or .venv
zip:
	@rm -f $(ZIP)
	@{ git ls-files --cached --others --exclude-standard; find results -type f 2>/dev/null; ls slides/*.pdf 2>/dev/null; } \
	  | grep -v -e '^data/' -e '^\.venv/' | sort -u \
	  | while read -r f; do [ -f "$$f" ] && echo "$$f"; done \
	  | zip -q -@ $(ZIP)
	@echo "wrote $(ZIP)"; unzip -l $(ZIP) | tail -1

clean:
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__ $(ZIP)
	rm -f data/cache/panel_*.parquet
