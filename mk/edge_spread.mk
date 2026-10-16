UPGRADES += upgrade-edge_spread
upgrade-edge_spread: | $(STAMP)
	$(PYI) src/spread_edge.py $(CONFIG)
