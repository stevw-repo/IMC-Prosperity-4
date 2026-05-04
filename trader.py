from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json
import math


class Trader:
    """
    IMC Prosperity 4 – Round 5 Algorithm  (v3)
    ============================================
    Fixes from v2:

    FIX-1  MR direction-aware target
           Root cause: MR always used desired=-POS_LIM on upward spike.
           For ROBOT_DISHES (trend +1, R²=0.765), spike up is TREND-ALIGNED.
           Going to -10 creates 20-unit churn identical to v1 SNACK bug.
           Fix: trend>0  → desired=0  (flatten, preserve micro-MR alpha)
                trend≤0  → desired=-POS_LIM  (contrarian short valid)
           Expected recovery: ~+20k (matches DISHES max DD of -20,219)

    FIX-2  Pairs H2_ONLY guard
           Root cause: pairs overlay ran in H1 on H2_ONLY products,
           adding ±PAIR_ALLOC to base=0, creating unintended H1 exposure.
           Fix: skip pairs leg for H2_ONLY products during H1.

    v2 changes retained:
      - SNACK_LEAD_LAG direction fix (corr sign + trend-alignment gate)
      - Adaptive MR threshold (max(MR_MIN_THRESH, MR_K × rolling_σ))
      - Additive pairs overlay (bounded by ±POS_LIM)
    """

    # ── Strategy flags ───────────────────────────────────────────────── #
    ENABLE_DIRECTIONAL_TREND = True
    ENABLE_PAIR_SPREAD       = True
    ENABLE_MR_ADAPTIVE       = True
    ENABLE_SNACK_LEAD_LAG    = True
    ENABLE_INTRADAY_TIMING   = True
    ENABLE_CLOSE_UNTRACKED   = True

    # ── Exchange constants ────────────────────────────────────────────── #
    POS_LIM = 10
    H1_END  = 500_000

    # ── Directional trend table ───────────────────────────────────────── #
    TREND: Dict[str, tuple] = {
        # HIGH-CONVICTION SHORTS  (R² ≥ 0.50)
        "MICROCHIP_OVAL":               (-1, 0.912),
        "UV_VISOR_AMBER":               (-1, 0.912),
        "PEBBLES_XS":                   (-1, 0.900),
        "PEBBLES_S":                    (-1, 0.798),
        "ROBOT_IRONING":                (-1, 0.766),
        "ROBOT_VACUUMING":              (-1, 0.764),
        "MICROCHIP_TRIANGLE":           (-1, 0.643),
        "ROBOT_LAUNDRY":                (-1, 0.629),
        "SNACKPACK_PISTACHIO":          (-1, 0.630),
        "MICROCHIP_RECTANGLE":          (-1, 0.571),
        "OXYGEN_SHAKE_MORNING_BREATH":  (-1, 0.556),
        "PANEL_1X4":                    (-1, 0.531),
        # MODERATE SHORTS  (0.30 ≤ R² < 0.50)
        "TRANSLATOR_ASTRO_BLACK":       (-1, 0.477),
        "SNACKPACK_CHOCOLATE":          (-1, 0.452),
        "TRANSLATOR_SPACE_GRAY":        (-1, 0.398),
        "PANEL_2X2":                    (-1, 0.360),
        # HIGH-CONVICTION LONGS  (R² ≥ 0.50)
        "OXYGEN_SHAKE_GARLIC":          (+1, 0.806),
        "SLEEP_POD_POLYESTER":          (+1, 0.802),
        "GALAXY_SOUNDS_BLACK_HOLES":    (+1, 0.785),
        "PANEL_2X4":                    (+1, 0.782),
        "ROBOT_DISHES":                 (+1, 0.765),
        "SNACKPACK_STRAWBERRY":         (+1, 0.708),
        "MICROCHIP_SQUARE":             (+1, 0.702),
        "PEBBLES_XL":                   (+1, 0.687),
        "SLEEP_POD_SUEDE":              (+1, 0.677),
        "UV_VISOR_MAGENTA":             (+1, 0.675),
        "SLEEP_POD_COTTON":             (+1, 0.645),
        "TRANSLATOR_VOID_BLUE":         (+1, 0.597),
        "ROBOT_MOPPING":                (+1, 0.523),
        "PEBBLES_M":                    (+1, 0.521),
        "UV_VISOR_RED":                 (+1, 0.485),
        # MODERATE LONGS  (0.30 ≤ R² < 0.50)
        "UV_VISOR_ORANGE":              (+1, 0.363),
        "OXYGEN_SHAKE_CHOCOLATE":       (+1, 0.354),
        "SLEEP_POD_NYLON":              (+1, 0.332),
    }

    # ── Intraday timing ───────────────────────────────────────────────── #
    H2_FLIP : set = {"SLEEP_POD_COTTON"}
    H1_ONLY : set = {"MICROCHIP_SQUARE"}
    H2_ONLY : set = {"ROBOT_DISHES", "UV_VISOR_MAGENTA", "GALAXY_SOUNDS_BLACK_HOLES"}

    # ── Adaptive MR ───────────────────────────────────────────────────── #
    MR_PRODUCT    = "ROBOT_DISHES"
    MR_K          = 2.5
    MR_WINDOW     = 50
    MR_HOLD       = 30
    MR_MIN_THRESH = 20

    # ── Pairs spread ──────────────────────────────────────────────────── #
    PAIR_A             = "ROBOT_DISHES"
    PAIR_B             = "ROBOT_IRONING"
    PAIR_BETA_INIT     = 0.72
    PAIR_BETA_LR       = 0.001
    PAIR_BETA_LO       = 0.30
    PAIR_BETA_HI       = 1.50
    PAIR_SPREAD_ENTRY  = 2.0
    PAIR_SPREAD_EXIT   = 0.5
    PAIR_ZSCORE_WINDOW = 200
    PAIR_ZSCORE_WARMUP = 30
    PAIR_ALLOC         = 5

    # ── Snack lead-lag ────────────────────────────────────────────────── #
    SNACK_LEADER  = "SNACKPACK_RASPBERRY"
    SNACK_THRESH  = 8
    SNACK_HOLD    = 3
    # corr: sign of follower's price response to a RASPBERRY price move
    # SNACKPACK_VANILLA: add once correlation sign confirmed from its price chart
    SNACK_FOLLOWERS: Dict[str, int] = {
        "SNACKPACK_PISTACHIO":  -1,
        "SNACKPACK_STRAWBERRY": -1,
    }

    # ================================================================== #
    #  HELPERS                                                             #
    # ================================================================== #

    @staticmethod
    def _bb(od: OrderDepth) -> Optional[int]:
        return max(od.buy_orders) if od.buy_orders else None

    @staticmethod
    def _ba(od: OrderDepth) -> Optional[int]:
        return min(od.sell_orders) if od.sell_orders else None

    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)

    def _order(
        self, product: str, target: int, cur: int, od: OrderDepth
    ) -> Optional[Order]:
        delta = target - cur
        if delta == 0:
            return None
        if delta > 0:
            ba = self._ba(od)
            if ba is None:
                return None
            qty = min(delta, self.POS_LIM - cur)
            return Order(product, ba, qty) if qty > 0 else None
        else:
            bb = self._bb(od)
            if bb is None:
                return None
            qty = min(-delta, cur + self.POS_LIM)
            return Order(product, bb, -qty) if qty > 0 else None

    @staticmethod
    def _rolling_std(data: list) -> float:
        n = len(data)
        if n < 2:
            return 0.0
        mean = sum(data) / n
        return math.sqrt(sum((x - mean) ** 2 for x in data) / (n - 1))

    # ================================================================== #
    #  MAIN RUN                                                            #
    # ================================================================== #

    def run(self, state: TradingState):

        # ── Restore state ─────────────────────────────────────────────── #
        try:
            sd = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            sd = {}

        prev_mid:    Dict[str, float] = sd.get("p",  {})
        mr_ticks:    Dict[str, int]   = sd.get("m",  {})
        snack_sig:   Dict[str, list]  = sd.get("s",  {})
        mr_diffs:    list             = sd.get("d",  [])
        pair_spread: list             = sd.get("ps", [])
        pair_pos:    int              = sd.get("pp",  0)
        pair_beta:   float            = sd.get("pb", self.PAIR_BETA_INIT)

        ts    = state.timestamp
        is_h2 = (ts >= self.H1_END)

        # ── Mid prices ────────────────────────────────────────────────── #
        cur_mid: Dict[str, float] = {}
        for p, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                cur_mid[p] = m

        desired: Dict[str, int] = {}

        # ── 1. Directional Trend ──────────────────────────────────────── #
        if self.ENABLE_DIRECTIONAL_TREND:
            for p, (direction, _r2) in self.TREND.items():
                if p not in state.order_depths:
                    continue
                eff = direction
                if self.ENABLE_INTRADAY_TIMING:
                    if is_h2:
                        if p in self.H2_FLIP:
                            eff = -direction
                        elif p in self.H1_ONLY:
                            eff = 0
                    else:
                        if p in self.H2_ONLY:
                            eff = 0
                desired[p] = eff * self.POS_LIM

        # ── 2. Pairs Spread ───────────────────────────────────────────── #
        if self.ENABLE_PAIR_SPREAD:
            mid_a = cur_mid.get(self.PAIR_A)
            mid_b = cur_mid.get(self.PAIR_B)

            if mid_a is not None and mid_b is not None:

                # Slow online β update
                if self.PAIR_A in prev_mid and self.PAIR_B in prev_mid:
                    da = mid_a - prev_mid[self.PAIR_A]
                    db = mid_b - prev_mid[self.PAIR_B]
                    if abs(db) > 0.5:
                        raw_beta = da / db
                        pair_beta = ((1.0 - self.PAIR_BETA_LR) * pair_beta
                                     + self.PAIR_BETA_LR * raw_beta)
                        pair_beta = max(self.PAIR_BETA_LO,
                                        min(self.PAIR_BETA_HI, pair_beta))

                spread = mid_a - pair_beta * mid_b
                pair_spread.append(spread)
                if len(pair_spread) > self.PAIR_ZSCORE_WINDOW:
                    pair_spread = pair_spread[-self.PAIR_ZSCORE_WINDOW:]

                if len(pair_spread) >= self.PAIR_ZSCORE_WARMUP:
                    mu  = sum(pair_spread) / len(pair_spread)
                    std = self._rolling_std(pair_spread)
                    z   = (spread - mu) / std if std > 0.1 else 0.0

                    if pair_pos == 0:
                        if z >  self.PAIR_SPREAD_ENTRY:
                            pair_pos = -1
                        elif z < -self.PAIR_SPREAD_ENTRY:
                            pair_pos =  1
                    elif pair_pos ==  1 and z > -self.PAIR_SPREAD_EXIT:
                        pair_pos = 0
                    elif pair_pos == -1 and z <  self.PAIR_SPREAD_EXIT:
                        pair_pos = 0

                    if pair_pos != 0:
                        base_a = desired.get(self.PAIR_A, 0)
                        base_b = desired.get(self.PAIR_B, 0)

                        # FIX-2: respect H2_ONLY restriction per leg
                        # Prevents pairs from creating H1 exposure on ROBOT_DISHES
                        a_suppressed = (self.PAIR_A in self.H2_ONLY and not is_h2)
                        b_suppressed = (self.PAIR_B in self.H2_ONLY and not is_h2)

                        if not a_suppressed:
                            desired[self.PAIR_A] = max(
                                -self.POS_LIM,
                                min(self.POS_LIM,
                                    base_a + pair_pos * self.PAIR_ALLOC)
                            )
                        if not b_suppressed:
                            desired[self.PAIR_B] = max(
                                -self.POS_LIM,
                                min(self.POS_LIM,
                                    base_b - pair_pos * self.PAIR_ALLOC)
                            )

        # ── 3. Adaptive MR ────────────────────────────────────────────── #
        if self.ENABLE_MR_ADAPTIVE:
            p = self.MR_PRODUCT
            if p in cur_mid and p in prev_mid:
                dp = cur_mid[p] - prev_mid[p]
                mr_diffs.append(dp)
                if len(mr_diffs) > self.MR_WINDOW:
                    mr_diffs = mr_diffs[-self.MR_WINDOW:]

                rolling_sigma = self._rolling_std(mr_diffs)
                threshold     = max(self.MR_MIN_THRESH,
                                    self.MR_K * rolling_sigma)

                if dp >= threshold:
                    mr_ticks[p] = self.MR_HOLD

            if p in mr_ticks:
                trend_dir = self.TREND.get(p, (0,))[0]

                # FIX-1: direction-aware MR target
                #
                # trend_dir > 0 (e.g. ROBOT_DISHES, R²=0.765):
                #   Spike UP is trend-aligned. The price is moving WITH its
                #   structural direction. Going to -10 reverses a +10 trend
                #   position → 20-unit churn → losses as trend continues.
                #
                #   target = 0: flatten at spike peak.
                #   When price reverts (micro mean-reversion), re-buy at
                #   lower price → still captures the spread profitably.
                #   No counter-trend exposure, no directional reversal.
                #
                # trend_dir ≤ 0 (no trend or downtrend):
                #   Spike UP is counter-trend. Contrarian short is valid.
                #   target = -POS_LIM: original behaviour, unchanged.
                if trend_dir > 0:
                    desired[p] = 0               # flatten only
                else:
                    desired[p] = -self.POS_LIM   # contrarian short

        # ── 4. Snack Lead-Lag v2 ──────────────────────────────────────── #
        if self.ENABLE_SNACK_LEAD_LAG:
            leader = self.SNACK_LEADER
            if leader in cur_mid and leader in prev_mid:
                dp = cur_mid[leader] - prev_mid[leader]

                if abs(dp) >= self.SNACK_THRESH:
                    rasp_dir = 1 if dp > 0 else -1

                    for fp, corr in self.SNACK_FOLLOWERS.items():
                        if fp not in state.order_depths:
                            continue
                        pred     = corr * rasp_dir
                        fp_trend = self.TREND.get(fp, (0,))[0]
                        if fp_trend == 0 or pred != fp_trend:
                            continue   # gate: only fire if signal == trend dir
                        snack_sig[fp] = [pred, self.SNACK_HOLD]

            for fp, (sig, tl) in list(snack_sig.items()):
                if fp not in state.order_depths:
                    continue
                fp_trend = self.TREND.get(fp, (0,))[0]
                if fp_trend == 0 or sig == fp_trend:
                    desired[fp] = sig * self.POS_LIM

        # ── 5. Generate orders ────────────────────────────────────────── #
        result: Dict[str, List[Order]] = {}

        for p, tgt in desired.items():
            if p not in state.order_depths:
                continue
            od  = state.order_depths[p]
            cur = state.position.get(p, 0)
            tgt = max(-self.POS_LIM, min(self.POS_LIM, tgt))
            o   = self._order(p, tgt, cur, od)
            if o:
                result[p] = [o]

        # Flatten orphaned positions
        if self.ENABLE_CLOSE_UNTRACKED:
            for p, cur in state.position.items():
                if cur != 0 and p not in desired:
                    if p in state.order_depths:
                        o = self._order(p, 0, cur, state.order_depths[p])
                        if o:
                            result[p] = [o]

        # ── 6. Serialise state ────────────────────────────────────────── #
        new_mr_ticks = {p: t - 1 for p, t in mr_ticks.items() if t > 1}
        new_snack    = {p: [d, t - 1] for p, (d, t) in snack_sig.items() if t > 1}

        tracked = (
            set(self.TREND)
            | {self.SNACK_LEADER, self.PAIR_A, self.PAIR_B, self.MR_PRODUCT}
            | set(self.SNACK_FOLLOWERS)
        )
        new_prev = {p: cur_mid[p] for p in tracked if p in cur_mid}

        trader_data = json.dumps({
            "p":  new_prev,
            "m":  new_mr_ticks,
            "s":  new_snack,
            "d":  mr_diffs,
            "ps": pair_spread,
            "pp": pair_pos,
            "pb": round(pair_beta, 6),
        }, separators=(",", ":"))

        return result, 0, trader_data