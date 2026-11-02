from llm_intervention_target_risk import structural_moments
from llm_intervention_contract import Node, Action
from llm_intervention_scm import LinearGaussianSCM
import json
model=LinearGaussianSCM((Node("Y",(),"u"),), {"Y":1.})
model.heads["Y"].mean[0]=1e12
mean,cov=structural_moments(model,Action("q","observe",(),("Y",),1))
print(json.dumps({"mean":mean.tolist(),"actual_variance":float(cov[0,0]),"analytic_variance":1.,"purpose":"arithmetic cancellation probe, no scientific world/model fitting"}))
