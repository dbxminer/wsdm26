import pandas as pd

# 1. Define historic and current prices
p = {'C': 2.00, 'P': 2.00, 'Ch': 3.00, 'N': 4.00, 'W': 1.50}
p_prime = {'C': 1.60, 'P': 2.00, 'Ch': 3.00, 'N': 4.00, 'W': 1.65}

# 2. Define cross-price elasticity matrix (symmetric)
E = {
    ('C', 'C'): -1.2, ('C', 'P'): 1.5,  ('C', 'Ch'): -0.4, ('C', 'N'): -0.2, ('C', 'W'): 0,
    ('P', 'C'): 1.5,  ('P', 'P'): -1.1, ('P', 'Ch'): -0.1, ('P', 'N'): -0.05,('P', 'W'): 0.1,
    ('Ch','C'): -0.4, ('Ch','P'): -0.1, ('Ch','Ch'): -0.9, ('Ch','N'): -0.1, ('Ch','W'): 0,
    ('N', 'C'): -0.2, ('N', 'P'): -0.05,('N', 'Ch'): -0.1, ('N', 'N'): -0.8, ('N', 'W'): 0,
    ('W', 'C'): 0,    ('W', 'P'): 0.1,  ('W', 'Ch'): 0,    ('W', 'N'): 0,    ('W', 'W'): -0.8
}

# 3. Define transactions (quantities)
transactions = [
    {'C':1, 'Ch':1},                      # T1
    {'C':1, 'Ch':1, 'N':1},               # T2
    {'P':1, 'Ch':1},                      # T3
    {'P':1, 'N':1},                       # T4
    {'C':1, 'P':1, 'N':1},                # T5
    {'W':2, 'Ch':1}                       # T6
]

# 4. Compute base demand q_B
q = {item: 0 for item in p}
for t in transactions:
    for item, qty in t.items():
        q[item] += qty

# 5. Compute effective demand ED_B
delta_p = {a: (p_prime[a] - p[a]) / p[a] for a in p}
ED = {}
for B in p:
    base = q[B]
    contributions = sum(E[(B, A)] * delta_p[A] * base for A in delta_p)
    ED[B] = base + contributions

# 6. Define itemsets to evaluate
itemsets = [
    ('C','Ch'), ('P','Ch'), ('C','N'),
    ('P','N'), ('W','Ch'), ('C','Ch','N')
]

# 7. For each itemset, compute historic utility TU(X), effective EU(X)
results = []
for X in itemsets:
    # historic utility TU(X)
    TU = sum(sum(p[a] * t[a] for a in X) for t in transactions if all(a in t for a in X))
    # effective demand ED(X) = min ED[a] for a in X
    ED_X = min(ED[a] for a in X)
    # current bundle price
    bundle_price = sum(p_prime[a] for a in X)
    # effective utility
    EU = ED_X * bundle_price
    results.append({
        'Itemset': ' {' + ','.join(X) + '} ',
        'Historic Utility': TU,
        'Effective Demand': round(ED_X, 2),
        'Bundle Price': round(bundle_price, 2),
        'Effective Utility': round(EU, 2)
    })

# 8. Display results
df = pd.DataFrame(results)
print(df.to_string(index=False))