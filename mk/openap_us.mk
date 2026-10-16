UPGRADES += upgrade-openap_us
upgrade-openap_us: | $(STAMP)
	$(PYI) src/openap_us.py $(CONFIG)
