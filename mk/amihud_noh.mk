UPGRADES += upgrade-amihud_noh
upgrade-amihud_noh: | $(STAMP)
	$(PYI) src/amihud_noh.py $(CONFIG)
