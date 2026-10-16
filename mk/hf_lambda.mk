UPGRADES += upgrade-hf_lambda
upgrade-hf_lambda: | $(STAMP)
	$(PYI) src/hf_lambda.py $(CONFIG)
