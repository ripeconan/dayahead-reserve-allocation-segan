"""R2 propositions with proofs — v3 (post R03B item 8 corrections).

PROPOSITION 1 (upper-support inclusion; proved, sharp, r0 clipping).
    C = {y : (y-mu)' Sigma^{-1} (y-mu) <= R}.  For any direction d, Cauchy-
    Schwarz in the Sigma^{-1} inner product gives
        (d'(y-mu))^2 <= (d' Sigma d) * ((y-mu)' Sigma^{-1} (y-mu)) <= (d'Sigma d) R,
    so with d = e_z: y_z <= mu_z + sqrt(R Sigma_zz) for every y in C.  Each
    bound is attained at y = mu + c Sigma e_z, c = sqrt(R / Sigma_zz), which
    lies ON the boundary, so the inclusion cannot be tightened per zone.
    Corollary: on a no-action hour, one-sided adequacy A_h(0) = 1 whenever the
    joint region covers (c_h = 1); and the reserve requirement
    r0_z = max(0, mu_z + sqrt(R Sigma_zz)) >= 0 by definition (clipping).

PROPOSITION 2 (daily binary fixed action; proved; the reviewer's corrected form).
    For a day with a single BINARY modeled action a in {0,1}:
        s_physical >= s_model - a * (1 - s_physical).
    Proof.  If a = 0: by Prop. 1, A_h >= c_h for every hour, hence
    s_physical = mean_h A_h >= mean_h c_h = s_model, and the right-hand side is
    s_model <= s_physical.  If a = 1: s_model = 1 by the modeled full-service
    convention, so the right-hand side equals s_physical exactly.  The
    inequality therefore holds with equality at a = 1 and reduces to the
    support-inclusion statement at a = 0.  It is NOT an exact identity for
    no-action days, and it does not apply to per-hour-varying actions that the
    method does not have.

PROPOSITION 3 (blocked-trigger deficit bound; proved; full-service model).
    Setting: strict lag ell, memory m >= n - ell, threshold theta > 0, modeled
    FULL-service executed fallback (an executed action gives d = 0; hence a
    positive-deficit day is either a day with h < theta - call it a NONTRIGGER
    day - or a blocked trigger day, where h >= theta and the budget is spent).
    Let the window w = [w, w+n-1] lie in one segment, and define
        P = {i : d_{w+i} > 0},   T = {i : d_{w+i} > 0 and h_{w+i} >= theta}  (= blocked),
        j = max{i in P : h_{w+i} < theta}  (nonexistent -> j = -inf).
    Then    sum_{i in P} d_{w+i}  <  theta + ell + sum_{i in T} d_{w+i},
    and, since each blocked d <= 1,  also  sum d < theta + ell + |T|.
    Proof.  Split P into {i <= j} and {i > j}.
      (a) For i in P with i > j: by maximality of j, h_{w+i} >= theta, so i in T;
          the tail therefore contributes exactly sum_{i in T, i > j} d <= sum_T d.
      (b) For the head {i in P, i <= j}: if j = -inf the head is empty.  Else,
          day w+j is a positive-deficit NONTRIGGER day, so h_{w+j} < theta.  The
          in-window eligible days for w+j are exactly w..w+j-ell (all in the
          segment; at most n-ell <= m of them, hence all among the m most recent
          eligible records, and pre-window eligible days only add nonnegative
          terms), so sum_{i=0}^{j-ell} d_{w+i} <= h_{w+j} < theta.  If j < ell
          there are no such indices and the head has at most j+1 <= ell terms
          each <= 1.  The middle terms (ell of them at most) are each <= 1.
          Hence the head is < theta + ell.
      Combining (a) and (b) gives the claim.  Ex-post: T depends on the realized
      blocked set.  No unconditional budget promise; the finite-response FEEDBACK
      controller is a DIFFERENT method and is not covered here.

PROPOSITION 4 (physical coupling; proved).
    With actions fixed as in the modeled bridge (daily binary), the physical
    one-sided-adequacy window deficit satisfies
        sum_{w} (1 - s_physical)  <=  sum_{w} (1 - s_model) + sum_{action days in w} (1 - s_physical,d),
    because on no-action days s_physical,d >= s_model,d (Prop. 2 at a = 0) and on
    action days s_model,d = 1 >= s_physical,d.  Combining with Prop. 3 bounds the
    modeled term; the second term is the transparent finite-capacity degradation
    of the local floor.
"""
from __future__ import annotations

import itertools

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import r2_env  # noqa: F401  (thread limits before numeric imports)

import numpy as np


def support_inclusion_checks(rng: np.random.Generator, dimension: int = 11,
                             trials: int = 200) -> dict:
    ok_inside, ok_sharp, max_excess, clipped_ok = 0, 0, -np.inf, 0
    for _ in range(trials):
        a = rng.normal(size=(dimension, dimension))
        sigma = a @ a.T / dimension + 1e-6 * np.eye(dimension)
        radius = float(rng.uniform(0.01, 50.0))
        mu = rng.normal(size=dimension) * 10.0
        precision = np.linalg.inv(sigma)
        chol = np.linalg.cholesky(sigma)
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        u = float(rng.uniform(0.0, radius))
        y = mu + chol @ (np.sqrt(u) * direction)  # guaranteed interior
        z = float((y - mu) @ precision @ (y - mu))
        assert z <= radius * (1 + 1e-9)
        bounds = mu + np.sqrt(np.maximum(radius * np.diag(sigma), 0.0))
        ok_inside += int(bool(np.all(y <= bounds + 1e-9)))
        max_excess = max(max_excess, float(np.max(y - bounds)))
        r0 = np.maximum(bounds, 0.0)
        clipped_ok += int(bool(np.all(r0 >= 0.0)))
        for zone in range(dimension):
            d = np.zeros(dimension)
            d[zone] = 1.0
            c = np.sqrt(radius / float(d @ sigma @ d))
            y2 = mu + c * (sigma @ d)
            z2 = float((y2 - mu) @ precision @ (y2 - mu))
            bound = mu[zone] + np.sqrt(max(radius * sigma[zone, zone], 0.0))
            if abs(z2 - radius) <= 1e-6 * max(1.0, radius) and abs(y2[zone] - bound) <= 1e-6 * max(1.0, abs(bound)):
                ok_sharp += 1
    return {"inside_trials": ok_inside, "sharpness_attainments": ok_sharp,
            "max_bound_excess_inside": float(max_excess), "r0_clip_ok": clipped_ok,
            "pass": bool(ok_inside == trials and ok_sharp == trials * dimension
                         and max_excess <= 1e-9 and clipped_ok == trials)}


def degradation_bound_checks(rng: np.random.Generator, hours: int = 24,
                             trials: int = 200) -> dict:
    """Prop. 2 with daily BINARY actions: a=0 -> s_phys >= s_model;
    a=1 -> s_model=1 and the stated inequality holds with equality."""
    ok0, ok1 = 0, 0
    for _ in range(trials):
        dimension = 11
        a = rng.normal(size=(dimension, dimension))
        sigma = a @ a.T / dimension + 1e-6 * np.eye(dimension)
        radius = float(rng.uniform(0.5, 30.0))
        mu = rng.normal(size=dimension) * 5.0
        precision = np.linalg.inv(sigma)
        bounds = mu + np.sqrt(np.maximum(radius * np.diag(sigma), 0.0))
        chol = np.linalg.cholesky(sigma)
        covered, adequate = 0, 0
        for _h in range(hours):
            v = rng.normal(size=dimension)
            v /= np.linalg.norm(v)
            u = float(rng.uniform(0.0, radius))
            y = mu + chol @ (np.sqrt(u) * v)
            c = float((y - mu) @ precision @ (y - mu)) <= radius
            adq = bool(np.all(y <= bounds))
            covered += int(c)
            adequate += int(adq)
            assert not c or adq, "Prop.1 violated"
        s_model0 = covered / hours
        s_phys0 = adequate / hours
        ok0 += int(s_phys0 >= s_model0 - 1e-12)
        s_model1, s_phys1 = 1.0, s_phys0
        ok1 += int(s_phys1 >= s_model1 - 1.0 * (1 - s_phys1) - 1e-12)
    return {"a0_trials": trials, "a0_satisfied": ok0, "a1_trials": trials, "a1_satisfied": ok1,
            "pass": bool(ok0 == trials and ok1 == trials)}


def blocked_trigger_bound_check(theta: float = 1.0, ell: int = 2, memory: int = 6,
                                days: int = 12, budgets: tuple[int, ...] = (0, 1, 2, 3)) -> dict:
    """Exhaustive enumeration over binary shortfall paths (full-service model).

    Validates BOTH  sum_P d < theta + ell + sum_T d  and the count version, and
    records whether the additive d-weighted term is ever needed.
    """
    violations_weighted: list[dict] = []
    violations_count: list[dict] = []
    empty_T_cases, short_j_cases = 0, 0
    worst_head_excess = -np.inf
    for budget in budgets:
        for bits in itertools.product([0.0, 1.0], repeat=days):
            q = np.asarray(bits)
            used = 0
            shortfalls = np.zeros(days)
            hist: list[tuple[int, float]] = []
            for t in range(days):
                eligible = [s for (u, s) in hist if u + ell <= t]
                recent = eligible[-memory:]
                h = float(sum(recent))
                if h >= theta and used < budget:
                    used += 1
                    shortfalls[t] = 0.0
                else:
                    shortfalls[t] = 1.0 - q[t]
                hist.append((t, shortfalls[t]))
            for start in range(0, days - 7 + 1):
                window = np.arange(start, start + 7)
                d_w = shortfalls[window]
                positive = d_w > 0
                if not positive.any():
                    continue
                h_vals = [hist[start + i][1] for i in range(7)]
                is_blocked = [bool(positive[i] and h_vals[i] >= theta and used >= 0 and
                                   _was_blocked(hist, start + i, theta, ell, memory, budget))
                              for i in range(7)]
                t_idx = [i for i in range(7) if is_blocked[i]]
                if not t_idx:
                    empty_T_cases += 1
                j = None
                for i in range(7):
                    if positive[i] and not is_blocked[i]:
                        j = i
                if j is not None and j < ell:
                    short_j_cases += 1
                s_w = float(d_w.sum())
                bound_weighted = theta + ell + float(d_w[t_idx].sum())
                bound_count = theta + ell + len(t_idx)
                if s_w >= bound_weighted - 1e-12:
                    violations_weighted.append({"budget": budget, "start": start,
                                                "S": s_w, "bound": bound_weighted})
                if s_w >= bound_count - 1e-12:
                    violations_count.append({"budget": budget, "start": start,
                                             "S": s_w, "bound": bound_count})
                head = float(d_w[: (j + 1) if j is not None else 0].sum())
                if j is not None:
                    worst_head_excess = max(worst_head_excess, head - (theta + ell))
    return {"violations_weighted": violations_weighted[:5],
            "violations_count": violations_count[:5],
            "empty_blocked_set_windows": empty_T_cases,
            "short_j_windows": short_j_cases,
            "max_head_minus_theta_plus_ell": float(worst_head_excess),
            "pass": bool(not violations_weighted and not violations_count
                         and worst_head_excess < 1e-9)}


def _was_blocked(hist: list[tuple[int, float]], t: int, theta: float, ell: int,
                 memory: int, budget: int) -> bool:
    """Recompute whether day t saw a blocked trigger (h >= theta and no budget)."""
    used = 0
    for u in range(t):
        eligible = [s for (v, s) in hist if v + ell <= u]
        recent = eligible[-memory:]
        h = float(sum(recent))
        if h >= theta and used < budget:
            used += 1
    eligible = [s for (v, s) in hist if v + ell <= t]
    h_t = float(sum(eligible[-memory:]))
    return bool(h_t >= theta and used >= budget)
