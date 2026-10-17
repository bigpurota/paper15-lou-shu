UPGRADES += upgrade-hf_lambda
HF_ARGS ?=
upgrade-hf_lambda: | $(STAMP)
	$(PYI) src/hf_lambda.py $(CONFIG) $(HF_ARGS)
