"""Find relations with best correlation at beta=1.0, layer=5"""
import sys
import numpy as np
from src.data import RelationDataset
from src.models import ModelAndTokenizer
from src.sweeps import sweep_relation

print("Loading model (GPT-J)...", flush=True)
mt = ModelAndTokenizer("gpt-j")
print("Model loaded\n", flush=True)

dataset = RelationDataset()
print(f"Testing {len(dataset)} relations at layer=5, beta=1.0\n", flush=True)

results = []
for i, relation in enumerate(dataset):
    relation_name = relation.relation_name
    try:
        print(f"[{i+1}/{len(dataset)}] {relation_name}...", end=" ", flush=True)
        
        result = sweep_relation(
            mt=mt,
            relation=relation,
            layers=[5],
            betas=[1.0],
            trials=3,
            n_train=5,
            batch_size=4,
            verbose=False
        )
        
        best = result.best_by_faithfulness()
        faith = best.faithfulness
        eff = best.efficacy
        
        # Calculate correlation between faithfulness and efficacy
        # (though with beta=1.0 fixed, we're really looking at the product)
        correlation = np.corrcoef([faith], [eff])[0, 1] if faith > 0 and eff > 0 else 0.0
        score = faith * eff
        
        results.append({
            'relation': relation_name,
            'faithfulness': faith,
            'efficacy': eff,
            'correlation': correlation,
            'score': score
        })
        
        print(f"faith={faith:.3f}, eff={eff:.3f}, score={score:.3f}", flush=True)
        
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

print("\n" + "="*80, flush=True)
print("TOP 20 RELATIONS (sorted by combined score: faithfulness × efficacy)", flush=True)
print("="*80, flush=True)
print(f"{'Rank':<6} {'Relation':<45} {'Faith':<8} {'Effic':<8} {'Score':<8}", flush=True)
print("-"*80, flush=True)

results.sort(key=lambda x: x['score'], reverse=True)
for i, r in enumerate(results[:20], 1):
    print(f"{i:<6} {r['relation']:<45} {r['faithfulness']:<8.3f} {r['efficacy']:<8.3f} {r['score']:<8.3f}", flush=True)

print("\n" + "="*80, flush=True)
print(f"Total relations tested: {len(results)}", flush=True)
print("="*80, flush=True)
