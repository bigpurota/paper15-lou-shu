UPGRADES += upgrade-moex
upgrade-moex: | $(STAMP)
	$(PYI) src/run_moex.py $(CONFIG)
