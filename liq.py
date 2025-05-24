import time
import math
import csv
from collections import defaultdict
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import random

# ---------------- Part 1: Parse and load transactions ----------------
def parse_transaction(line):
    parts = line.strip().split(':', 2)
    if len(parts) != 3:
        raise ValueError(f"Bad transaction format: {line!r}")
    items_str, total_str, indiv_str = parts
    items = list(map(int, items_str.split()))
    indiv_profits = list(map(float, indiv_str.split()))
    if len(indiv_profits) != len(items):
        raise AssertionError(
            f"profit count ({len(indiv_profits)}) ≠ item count ({len(items)})"
        )
    return items, indiv_profits

# ---------------- Part 2: Single scan for min-price and frequency ----------------
def scan_prices_and_freq(transactions):
    price = {}
    freq  = defaultdict(int)
    for items, profits in tqdm(transactions, desc="Scanning prices & freq", unit="txn"):
        for item, p in zip(items, profits):
            freq[item] += 1
            if item not in price or p < price[item]:
                price[item] = p
    return price, freq

# ---------------- Price adjustment ----------------
def adjust_prices(price, x_percent, y_percent, alpha=2):
    """
    Reprice a random x% of items by a heavy-tailed Δ in [-y%, +y%],
    using Python’s continuous Pareto variate + random sign.
    """
    items = list(price.keys())
    k     = max(1, int(len(items) * x_percent / 100))
    sampled = random.sample(items, k)

    # 1) Draw k Pareto variates (min 1.0)
    raws = [random.paretovariate(alpha) for _ in sampled]
    max_raw = max(raws)

    # 2) Normalize mags to [0,1] and scale to y%
    mags = [(r / max_raw) * (y_percent / 100) for r in raws]

    # 3) Random ± sign and apply
    for item, mag in zip(sampled, mags):
        sign = 1 if random.random() < 0.5 else -1
        price[item] *= (1 + sign * mag)

    return sampled

# ---------------- Elasticity matrix loader (dynamic alpha sign) ----------------
def load_cross_price(distance_file, alpha=6, beta=0.6):
    with open(distance_file, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header_ids = [int(x) for x in rows[0][1:]]
    all_dists = []
    for row in rows[1:]:
        all_dists.extend(float(x) for x in row[1:])
    all_dists.sort()
    m = len(all_dists)
    threshold = (all_dists[m//2] if m%2==1 else
                 (all_dists[m//2-1] + all_dists[m//2]) / 2)

    cross_price = {}
    for row in rows[1:]:
        B = int(row[0])
        dists = [float(x) for x in row[1:]]
        cross_price[B] = {}
        for A, dist in zip(header_ids, dists):
            if A == B: continue
            sign = 1.0 if dist <= threshold else -1.0
            cross_price[B][A] = sign * abs(alpha) * math.exp(-beta * dist)

    return cross_price

# ---------------- Compute effective demand per item ----------------
def compute_effective_demand(freq, repriced, orig_price, new_price, cross_price):

    E_D = {}
    demand_change = {}
    for B, qB in freq.items():
        delta_sum = 0.0
        # Sum cross-price effects from all repriced items
        for A in repriced:
            pA0, pA1 = orig_price[A], new_price[A]
            if pA0 == 0:
                continue
            eps = cross_price.get(B, {}).get(A, 0.0)
            delta_sum += eps * ((pA1 - pA0) / pA0) * qB

        # Compute raw new demand and bound it to be non-negative
        E = qB + delta_sum
        E = max(E, 0.0)

        E_D[B] = E
        demand_change[B] = (E - qB) / qB if qB else 0.0

    return E_D, demand_change

# ---------------- Part 3: Top-K singletons ----------------
def level1_itemsets(utilities, k):
    return [item for item, _ in
            sorted(utilities.items(), key=lambda x: x[1], reverse=True)[:k]]

# ---------------- Part 4: Prune transactions ----------------
def prune_transactions(transactions, top_items_set):
    pruned = []
    for items, _ in tqdm(transactions, desc="Pruning transactions", unit="txn"):
        kept = sorted(i for i in items if i in top_items_set)
        if kept:
            pruned.append(tuple(kept))
    return pruned

# ---------------- Part 5: Level-wise itemset generation ----------------
def generate_next_level(prev_itemsets, pruned_transactions, level1_set):
    new_counts = defaultdict(int)
    prev_set   = set(prev_itemsets)
    for I in tqdm(prev_itemsets, desc="Generating next level", unit="iset"):
        hi = max(I)
        for trans in pruned_transactions:
            if set(I).issubset(trans):
                items = set(trans)
                for j in items:
                    if j > hi and j in level1_set:
                        C = tuple(sorted(I + (j,)))
                        new_counts[C] += 1
                lower = [j for j in items if j < hi and j in level1_set and j not in I]
                for j in lower:
                    TI = tuple(sorted((j,) + tuple(x for x in I if x != hi)))
                    if TI not in prev_set:
                        C = tuple(sorted(TI + (hi,)))
                        new_counts[C] += 1
    return dict(new_counts)

# ---------------- Main mining function (returns L, price, demand_change, E_D_map, freq) ----------------
def mine_itemsets(transactions,
                  cross_price,
                  approach='support',
                  top_k=500,
                  max_level=5,
                  x_percent=45,
                  y_percent=45):
    price, freq = scan_prices_and_freq(transactions)
    orig_price = price.copy()
    random.seed(42)
    repriced = adjust_prices(price, x_percent, y_percent)
    E_D_map, demand_change = compute_effective_demand(
        freq, repriced, orig_price, price, cross_price)

    # Level 1 utility
    util1 = {}
    for i, q in freq.items():
        util1[i] = price[i] * (q if approach=='support' else E_D_map.get(i, q))
    L = {1: { (i,): freq[i] for i in level1_itemsets(util1, top_k) }}
    level1_set = set(i for (i,) in L[1])
    pruned = prune_transactions(transactions, level1_set)
    prev = L[1]

    # Higher levels
    for n in range(2, max_level+1):
        curr = generate_next_level(prev, pruned, level1_set)
        if not curr: break
        utiln = {}
        for iset, supp in curr.items():
            p_sum = sum(price[i] for i in iset)
            if approach == 'support':
                utiln[iset] = p_sum * supp
            else:
                EDs = [E_D_map[B] * (supp / freq[B]) for B in iset]
                ED_z = min(EDs)
                utiln[iset] = p_sum * ED_z
        topn = sorted(utiln.items(), key=lambda x: x[1], reverse=True)[:top_k]
        pruned_curr = {iset: curr[iset] for iset, _ in topn}
        L[n] = pruned_curr
        prev = pruned_curr

    return L, price, demand_change, E_D_map, freq

# ---------------- Compute test utility for top-zeta itemsets ----------------
def compute_test_utility_top(L, price, demand_change, test_transactions,
                             approach, zeta, E_D_map, freq):
    test_price = {i: price[i] * (1 + demand_change.get(i, 0.0)) for i in price}

    # Flatten all itemsets into a utility map
    util_map = {}
    second_util_map={}
    for itemsets in L.values():
        for iset, supp in itemsets.items():
            p_sum = sum(price[i] for i in iset)
            if approach == 'support':
                util_map[iset] = p_sum * supp
            else:
                EDs = [E_D_map[B] * (supp / freq[B]) for B in iset]
                #print(EDs)
                util_map[iset] = p_sum * min(EDs)

    # Select top-zeta itemsets by training utility
    top_isets = sorted(util_map.items(), key=lambda x: x[1], reverse=True)[:zeta]

    # Sum up utilities in test fold: each time an itemset appears, add its price-sum
    total_util = 0.0
    for iset, _ in top_isets:
        pset = sum(test_price[i] for i in iset)
        sset = set(iset)
        for items, _ in test_transactions:
            if sset.issubset(items):
                total_util += pset

    return total_util

# ---------------- Experiment loops (varying lambda, x, y, alpha, beta, zeta) ----------------
if __name__ == "__main__":
    random.seed(42)
    # load data
    with open("liquor_15.txt") as f:
        raw = f.readlines()
    parsed = [parse_transaction(l) for l in raw]
    train, test = train_test_split(parsed, test_size=0.25, random_state=42)

    # default parameters
    lambdas = range(500, 501, 500)
    x_percents = [30,40,50,60,70]
    y_percents = [30,40,50,60,70]
    alpha_values = [1,2,3,4,5]
    beta_values = [0.1]
    zetas = range(100, 501, 100)
    defaults = dict(lam=500, x=50, y=50, alpha=3, beta=0.1, zeta=300)
    max_size = 5
    approaches = ['support', 'demand']

    # baseline experiments (vary lambda, x, y) with default alpha/beta/zeta
    cross_price = load_cross_price("distance_matrix_liquor.csv",
                                   defaults['alpha'], defaults['beta'])

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_results = defaultdict(lambda: defaultdict(list))

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(parsed)):
        train = [parsed[i] for i in train_idx]
        test = [parsed[i] for i in test_idx]
        for approach in approaches:
            print(f"=== Approach: {approach} ===")
            # vary x%
            for x in x_percents:
                start = time.time()
                L, price_map, demand_ch, E_D_map, freq = mine_itemsets(
                    train, cross_price, approach,
                    defaults['lam'], max_size, x, defaults['y'])
                dur = time.time() - start
                tu = compute_test_utility_top(
                    L, price_map, demand_ch, test,
                    approach, defaults['zeta'], E_D_map, freq)
                print(f"x%={x}: time={dur:.2f}s, test util={tu:.2f}")
        for approach in approaches:
            print(f"=== Approach: {approach} ===")
            # vary y%
            for y in y_percents:
                start = time.time()
                L, price_map, demand_ch, E_D_map, freq = mine_itemsets(
                    train, cross_price, approach,
                    defaults['lam'], max_size, defaults['x'], y)
                dur = time.time() - start
                tu = compute_test_utility_top(
                    L, price_map, demand_ch, test,
                    approach, defaults['zeta'], E_D_map, freq)
                print(f"y%={y}: time={dur:.2f}s, test util={tu:.2f}")
        # vary alpha
        for alpha in alpha_values:
            cross_price = load_cross_price("distance_matrix_liquor.csv", alpha, defaults['beta'])
            for approach in approaches:
                start = time.time()
                L, price_map, demand_ch, E_D_map, freq = mine_itemsets(
                    train, cross_price, approach,
                    defaults['lam'], max_size, defaults['x'], defaults['y'])
                dur = time.time() - start
                tu = compute_test_utility_top(
                    L, price_map, demand_ch, test,
                    approach, defaults['zeta'], E_D_map, freq)
                print(f"alpha={alpha}, approach={approach}: time={dur:.2f}s, test util={tu:.2f}")
        # vary beta
        for beta in beta_values:
            cross_price = load_cross_price("distance_matrix_liquor.csv", defaults['alpha'], beta)
            for approach in approaches:
                start = time.time()
                L, price_map, demand_ch, E_D_map, freq = mine_itemsets(
                    train, cross_price, approach,
                    defaults['lam'], max_size, defaults['x'], defaults['y'])
                dur = time.time() - start
                tu = compute_test_utility_top(
                    L, price_map, demand_ch, test,
                    approach, defaults['zeta'], E_D_map, freq)
                print(f"beta={beta}, approach={approach}: time={dur:.2f}s, test util={tu:.2f}")
        # vary zeta
        cross_price = load_cross_price("distance_matrix_liquor.csv",
                                       defaults['alpha'], defaults['beta'])
        for zeta in zetas:
            for approach in approaches:
                start = time.time()
                L, price_map, demand_ch, E_D_map, freq = mine_itemsets(
                    train, cross_price, approach,
                    defaults['lam'], max_size, defaults['x'], defaults['y'])
                dur = time.time() - start
                tu = compute_test_utility_top(
                    L, price_map, demand_ch, test,
                    approach, zeta, E_D_map, freq)
                print(f"zeta={zeta}, approach={approach}: time={dur:.2f}s, test util={tu:.2f}")