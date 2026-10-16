UPGRADES += upgrade-sentiment
upgrade-sentiment: | $(STAMP)
	$(PYI) src/sentiment.py $(CONFIG)
