"""Independent fractional-path checks of the proved deficit accounting bound."""
import json
from pathlib import Path
import random

rng=random.Random(20261011)
checks=0
violations=[]
for lag in (1,2,3,5):
    for width in (3,7,14):
        memory=max(1,width-lag)
        for theta in (.2,1.,2.5,6.):
            for budget in (0,1,3,10):
                for repetition in range(30):
                    q=[rng.random() if repetition%3 else rng.choice((0.,.5,1.)) for _ in range(30)]
                    deficits=[]; blocked=[]; actions=[]; used=0
                    for t in range(len(q)):
                        # Eligible records are source indices u <= t-lag.
                        due=deficits[:max(0,t-lag+1)]
                        state=sum(due[-memory:])
                        trigger=state>=theta
                        action=trigger and used<budget
                        block=trigger and not action
                        actions.append(action); blocked.append(block)
                        used+=int(action)
                        deficits.append(0. if action else 1.-q[t])
                    # Physical service can exceed joint coverage without an action;
                    # action-day finite adequacy can lie anywhere in[0,1].
                    physical=[rng.random() if a else qq+(1.-qq)*rng.random() for a,qq in zip(actions,q)]
                    for start in range(len(q)-width+1):
                        w=range(start,start+width)
                        modeled=sum(deficits[t] for t in w)
                        blocked_deficit=sum(deficits[t] for t in w if blocked[t])
                        model_bound=theta+lag+blocked_deficit
                        finite_term=sum(1.-physical[t] for t in w if actions[t])
                        phys_deficit=sum(1.-physical[t] for t in w)
                        checks+=1
                        if not modeled<model_bound or phys_deficit>modeled+finite_term+1e-12:
                            violations.append({'lag':lag,'width':width,'memory':memory,'theta':theta,'budget':budget,'start':start,
                                               'modeled':modeled,'model_bound':model_bound,'physical':phys_deficit,'finite_term':finite_term})
result={'synthetic_only':True,'windows_checked':checks,'violations':violations[:10],
        'pass':not violations,'reviewer_proof':'The last positive-deficit nontrigger has an eligible prefix sum <theta and at most lag remaining head terms. Later positive deficits are blocked; support inclusion transfers the bound with action-day physical deficit added. This numerical check supplements that proof.'}
path=Path(__file__).resolve().parent/'REVIEWER_R2_PROOF_RESULTS.json'
path.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
