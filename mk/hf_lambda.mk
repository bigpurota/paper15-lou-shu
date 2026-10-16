UPGRADES += upgrade-hf_lambda
# HF_ARGS: extra flags, e.g. `make upgrade-hf_lambda HF_ARGS="--no-download --strict-offline"`
# (offline from data/raw; --strict-offline stops if any needed file is missing instead of using a subset)
HF_ARGS ?=
upgrade-hf_lambda: | $(STAMP)
	$(PYI) src/hf_lambda.py $(CONFIG) $(HF_ARGS)
