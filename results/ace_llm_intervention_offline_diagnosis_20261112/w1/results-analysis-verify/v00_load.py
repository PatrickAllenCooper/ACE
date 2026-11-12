"""Verifier step 0: load the raw CSVs and describe their structure. Fresh code."""
import pandas as pd, numpy as np
D = "/Users/pat/code/ACE/results/ace_llm_intervention_scientific_finalization_20261108/reporting01/data01/"
cells = pd.read_csv(D + "cells.csv")
curves = pd.read_csv(D + "curves.csv")
print("cells shape", cells.shape)
print(cells.dtypes)
print("curves shape", curves.shape)
print(curves.dtypes)
print("policies:", cells.policy_id.unique().tolist())
print("worlds:", sorted(cells.world_id.unique().tolist()))
print("n worlds", cells.world_id.nunique())
print(cells.groupby(["cost_stratum", "graph_stratum", "regime"]).world_id.nunique())
print("termination reasons:", cells.termination_reason.value_counts().to_dict())
print("failed/unresolved:", cells.failed.sum(), cells.unresolved.sum())
print("curve points per policy:")
print(curves.groupby("policy_id").size())
print("curves action_id values:", curves.action_id.fillna("<NA>").value_counts().to_dict())
print("total_cost range", curves.total_cost_units.min(), curves.total_cost_units.max())
print("acq cost range", curves.acquisition_cost_units.min(), curves.acquisition_cost_units.max())
# world design table
w = cells.drop_duplicates("world_id")[["world_id", "cost_stratum", "graph_stratum", "regime"]].sort_values("world_id")
print(w.to_string(index=False))
