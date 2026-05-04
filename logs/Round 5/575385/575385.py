from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional
import json


class Trader:
    """
    IMC Prosperity 4 – Round 5 Algorithm
    ======================================
    Five independently togglable sub-strategies:

    1. DIRECTIONAL_TREND  – Maintain ±POS_LIM in the identified trend direction
                            for every product in the TREND table.
    2. MR_ROBOT_DISHES    – Contrarian mean-reversion on ROBOT_DISHES when a
                            tick-over-tick mid-price move exceeds MR_THRESH.
                            Overrides the directional long for MR_HOLD ticks.
    3. SNACK_LEAD_LAG     – Use large SNACKPACK_RASPBERRY moves to pre-position
                            in SNACKPACK_PISTACHIO and SNACKPACK_STRAWBERRY.
    4. INTRADAY_TIMING    – Adjust or zero positions based on H1 / H2 split
                            (timestamps 0-499,900 = H1; 500,000-999,900 = H2).
    5. CLOSE_UNTRACKED    – Flat any position that no active strategy covers
                            (safety net to prevent orphaned positions).

    Set the ENABLE_* flags to True/False to isolate individual strategies.
    """

    # ------------------------------------------------------------------ #
    #  STRATEGY ENABLE / DISABLE                                           #
    #  Set to False to isolate / benchmark individual sub-strategies.      #
    # ------------------------------------------------------------------ #
    ENABLE_DIRECTIONAL_TREND = True
    ENABLE_MR_ROBOT_DISHES   = True
    ENABLE_SNACK_LEAD_LAG    = True
    ENABLE_INTRADAY_TIMING   = True
    ENABLE_CLOSE_UNTRACKED   = True

    # ------------------------------------------------------------------ #
    #  EXCHANGE PARAMETERS                                                 #
    # ------------------------------------------------------------------ #
    POS_LIM = 10          # position limit for every product (Round 5 rules)
    H1_END  = 500_000     # intraday timestamp boundary between H1 and H2

    # ------------------------------------------------------------------ #
    #  DIRECTIONAL TREND TABLE                                             #
    #                                                                      #
    #  Format:  product_name -> (direction, r_squared)                     #
    #    direction  +1 = hold max long,  -1 = hold max short               #
    #    r_squared  OLS R² from per-product regression on sample Days 2-4  #
    #                                                                      #
    #  Inclusion criteria: R² >= 0.30  AND  |slope| >= 0.01 ticks/tick.   #
    #  To exclude a product entirely, remove its entry from this dict.     #
    #  To add a product, append (direction, r2) for any product name.      #
    # ------------------------------------------------------------------ #
    TREND: Dict[str, tuple] = {

        # ── HIGH-CONVICTION SHORTS  (R² ≥ 0.50) ──────────────────────── #
        # All show negative slope_consistent=True across Days 2-4 except
        # where annotated.
        "MICROCHIP_OVAL":               (-1, 0.912),  # -0.171/tick  -44.8%
        "PEBBLES_XS":                   (-1, 0.900),  # -0.159/tick  -39.6%  front-loaded
        "UV_VISOR_AMBER":               (-1, 0.912),  # -0.110/tick  -28.7%  [deceleration: Day3+]
        "PEBBLES_S":                    (-1, 0.798),  # -0.086/tick  -25.8%
        "ROBOT_IRONING":                (-1, 0.766),  # -0.078/tick  -23.4%
        "ROBOT_VACUUMING":              (-1, 0.764),  # -0.054/tick  -16.2%
        "MICROCHIP_TRIANGLE":           (-1, 0.643),  # -0.077/tick  -23.1%
        "ROBOT_LAUNDRY":                (-1, 0.629),  # -0.056/tick  -16.9%
        "SNACKPACK_PISTACHIO":          (-1, 0.630),  # -0.017/tick   -5.2%
        "MICROCHIP_RECTANGLE":          (-1, 0.571),  # -0.066/tick  -19.7%
        "OXYGEN_SHAKE_MORNING_BREATH":  (-1, 0.556),  # -0.056/tick  -16.9%
        "PANEL_1X4":                    (-1, 0.531),  # -0.070/tick  -21.1%
        "TRANSLATOR_ASTRO_BLACK":       (-1, 0.477),  # -0.039/tick  -11.7%
        "SNACKPACK_CHOCOLATE":          (-1, 0.452),  # -0.016/tick   -4.7%
        "TRANSLATOR_SPACE_GRAY":        (-1, 0.398),  # -0.037/tick  -11.0%
        "PANEL_2X2":                    (-1, 0.360),  # -0.047/tick  -14.0%

        # ── HIGH-CONVICTION LONGS  (R² ≥ 0.50) ───────────────────────── #
        "OXYGEN_SHAKE_GARLIC":          (+1, 0.806),  # +0.099/tick  +29.6%  consistent all days
        "SLEEP_POD_POLYESTER":          (+1, 0.802),  # +0.101/tick  +30.3%  [Day-4 reversal risk]
        "GALAXY_SOUNDS_BLACK_HOLES":    (+1, 0.785),  # +0.098/tick  +29.4%  back-loaded
        "PANEL_2X4":                    (+1, 0.782),  # +0.064/tick  +19.2%  consistent all days
        "ROBOT_DISHES":                 (+1, 0.765),  # +0.056/tick  +16.9%  [also MR strategy]
        "SNACKPACK_STRAWBERRY":         (+1, 0.708),  # +0.035/tick  +10.6%
        "MICROCHIP_SQUARE":             (+1, 0.702),  # +0.177/tick  +53.1%  [Day-4 reversal risk]
        "PEBBLES_XL":                   (+1, 0.687),  # +0.170/tick  +51.0%
        "SLEEP_POD_SUEDE":              (+1, 0.677),  # +0.086/tick  +25.7%
        "UV_VISOR_MAGENTA":             (+1, 0.675),  # +0.058/tick  +17.5%  [deceleration: Day3+]
        "SLEEP_POD_COTTON":             (+1, 0.645),  # +0.082/tick  +24.7%  H2-flip product
        "TRANSLATOR_VOID_BLUE":         (+1, 0.597),  # +0.052/tick  +15.5%
        "ROBOT_MOPPING":                (+1, 0.523),  # +0.064/tick  +19.2%
        "PEBBLES_M":                    (+1, 0.521),  # +0.057/tick  +17.2%
        "UV_VISOR_RED":                 (+1, 0.485),  # +0.047/tick  +14.2%

        # ── MODERATE LONGS  (0.30 ≤ R² < 0.50) ──────────────────────── #
        "UV_VISOR_ORANGE":              (+1, 0.363),  # +0.038/tick  +11.5%
        "OXYGEN_SHAKE_CHOCOLATE":       (+1, 0.354),  # +0.039/tick  +11.6%
        "SLEEP_POD_NYLON":              (+1, 0.332),  # +0.034/tick  +10.1%
    }

    # ------------------------------------------------------------------ #
    #  INTRADAY TIMING OVERRIDES                                           #
    #  Applied only when ENABLE_INTRADAY_TIMING = True.                   #
    #                                                                      #
    #  H2_FLIP   : trend reverses in H2  (H1 slope opposes H2 slope)     #
    #  H1_ONLY   : trend exhausted after H1  (near-zero H2 slope)        #
    #  H2_ONLY   : minimal H1 trend; enter only after H1_END             #
    # ------------------------------------------------------------------ #
    H2_FLIP  : set = {
        "SLEEP_POD_COTTON",           # H1=+0.109  H2=-0.056  (ratio=-0.51)
    }
    H1_ONLY  : set = {
        "MICROCHIP_SQUARE",           # H1=+0.317  H2=+0.005  (ratio=0.015)
    }
    H2_ONLY  : set = {
        "ROBOT_DISHES",               # H1=+0.016  H2=+0.102  (ratio=6.3)
        "UV_VISOR_MAGENTA",           # H1=+0.005  H2=+0.074  (ratio=16.0)
        "GALAXY_SOUNDS_BLACK_HOLES",  # H1=+0.074  H2=+0.235  (ratio=3.2)
    }

    # ------------------------------------------------------------------ #
    #  MEAN-REVERSION PARAMETERS – ROBOT_DISHES                           #
    #                                                                      #
    #  Analysis result (Alpha 4):                                          #
    #    |ΔP| ≥ 20  →  E[gross]=+12.5 ticks, E[net after spread]=+6.5   #
    #    win_rate = 0.409, n_events/day ≈ 58 over 3-day sample.          #
    #                                                                      #
    #  MR_THRESH raised to 50 vs the analysis threshold of 20 because    #
    #  the high-volatility regime (timestamp > 4.1M in sample = H2 of    #
    #  day) produces ±100-tick moves that would trigger too frequently    #
    #  at threshold 20, eroding profitability via spread costs.           #
    #  Lower MR_THRESH (e.g. 20) for higher signal frequency if the      #
    #  competition day has a low-volatility regime.                        #
    # ------------------------------------------------------------------ #
    MR_THRESH = 50    # min tick-over-tick ΔP to trigger contrarian short
    MR_HOLD   = 30    # ticks to maintain the contrarian position

    # ------------------------------------------------------------------ #
    #  SNACK PACK LEAD-LAG PARAMETERS                                      #
    #                                                                      #
    #  RASPBERRY Granger-causes PISTACHIO and STRAWBERRY (lag 1-2,        #
    #  p<0.0001 in all directions). A large RASPBERRY move predicts       #
    #  PISTACHIO/STRAWBERRY moves in the same direction within 1-3 ticks. #
    #  SNACK_THRESH = 8 ticks; tighten to reduce noise or loosen          #
    #  to increase signal frequency.                                       #
    # ------------------------------------------------------------------ #
    SNACK_THRESH    = 8      # RASPBERRY |ΔP| to trigger (ticks)
    SNACK_HOLD      = 3      # ticks to maintain follow position
    SNACK_LEADER    = "SNACKPACK_RASPBERRY"
    SNACK_FOLLOWERS = ("SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY")

    # ================================================================== #
    #  INTERNAL HELPERS                                                    #
    # ================================================================== #

    @staticmethod
    def _best_bid(od: OrderDepth) -> Optional[int]:
        """Highest price at which a bot is willing to buy."""
        return max(od.buy_orders) if od.buy_orders else None

    @staticmethod
    def _best_ask(od: OrderDepth) -> Optional[int]:
        """Lowest price at which a bot is willing to sell."""
        return min(od.sell_orders) if od.sell_orders else None

    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        """Mid price; falls back to single-side best if one side is empty."""
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        if bb is not None:
            return float(bb)
        if ba is not None:
            return float(ba)
        return None

    def _make_order(
        self,
        product:    str,
        target_pos: int,
        cur_pos:    int,
        od:         OrderDepth,
    ) -> Optional[Order]:
        """
        Construct a single aggressive (liquidity-taking) order to move
        cur_pos toward target_pos.  Respects POS_LIM on both sides.
        Returns None if no move is required or the book is empty.
        """
        delta = target_pos - cur_pos
        if delta == 0:
            return None

        if delta > 0:                                    # need to BUY
            ba = self._best_ask(od)
            if ba is None:
                return None
            qty = min(delta, self.POS_LIM - cur_pos)    # cannot exceed +POS_LIM
            return Order(product, ba, qty) if qty > 0 else None

        else:                                            # need to SELL
            bb = self._best_bid(od)
            if bb is None:
                return None
            qty = min(-delta, cur_pos + self.POS_LIM)   # cannot go below -POS_LIM
            return Order(product, bb, -qty) if qty > 0 else None

    # ================================================================== #
    #  MAIN ENTRY POINT                                                    #
    # ================================================================== #

    def run(self, state: TradingState):

        # ── 1. Restore persistent state ──────────────────────────────── #
        # traderData is a JSON string persisted from the previous tick.
        # Keys are shortened to save space (50 KB cap on traderData).
        #   "p"  previous mid prices          {product: float}
        #   "m"  MR ticks remaining           {product: int}
        #   "s"  snack signals                {product: [direction, ticks_left]}
        try:
            sd = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            sd = {}

        prev_mid:  Dict[str, float]  = sd.get("p", {})
        mr_ticks:  Dict[str, int]    = sd.get("m", {})
        snack_sig: Dict[str, list]   = sd.get("s", {})

        ts    = state.timestamp
        is_h2 = ts >= self.H1_END

        # ── 2. Compute current mid prices ────────────────────────────── #
        cur_mid: Dict[str, float] = {}
        for p, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                cur_mid[p] = m

        # ── 3. Determine desired position for every product ───────────── #
        desired: Dict[str, int] = {}

        # ── 3a. Directional Trend ──────────────────────────────────────  #
        if self.ENABLE_DIRECTIONAL_TREND:
            for p, (direction, r2) in self.TREND.items():
                if p not in state.order_depths:
                    continue

                eff = direction                       # effective direction this tick

                if self.ENABLE_INTRADAY_TIMING:
                    if is_h2:
                        if p in self.H2_FLIP:
                            eff = -direction          # reverse in second half
                        elif p in self.H1_ONLY:
                            eff = 0                   # trend gone; go flat
                    else:                             # H1
                        if p in self.H2_ONLY:
                            eff = 0                   # wait for H2 entry window

                desired[p] = eff * self.POS_LIM

        # ── 3b. Mean-Reversion Override – ROBOT_DISHES ────────────────── #
        if self.ENABLE_MR_ROBOT_DISHES:
            p = "ROBOT_DISHES"
            if p in cur_mid and p in prev_mid:
                dp = cur_mid[p] - prev_mid[p]
                if dp >= self.MR_THRESH:
                    # Large UP move: price likely to revert downward.
                    # Override directional long with contrarian short.
                    mr_ticks[p] = self.MR_HOLD

            if p in mr_ticks:
                # Actively in MR short mode – override whatever directional set.
                desired[p] = -self.POS_LIM

        # ── 3c. Snack Pack Lead-Lag ────────────────────────────────────── #
        if self.ENABLE_SNACK_LEAD_LAG:
            leader = self.SNACK_LEADER
            if leader in cur_mid and leader in prev_mid:
                dp = cur_mid[leader] - prev_mid[leader]
                if abs(dp) >= self.SNACK_THRESH:
                    sig = 1 if dp > 0 else -1
                    for fp in self.SNACK_FOLLOWERS:
                        if fp in state.order_depths:
                            # Reset timer each time a new signal fires.
                            snack_sig[fp] = [sig, self.SNACK_HOLD]

            # Apply any active snack signals (override directional).
            for fp, (sig, tl) in list(snack_sig.items()):
                if fp in state.order_depths:
                    desired[fp] = sig * self.POS_LIM

        # ── 4. Generate orders ────────────────────────────────────────── #
        result: Dict[str, List[Order]] = {}

        # 4a. Orders derived from desired-position map.
        for p, tgt in desired.items():
            if p not in state.order_depths:
                continue
            od  = state.order_depths[p]
            cur = state.position.get(p, 0)
            tgt = max(-self.POS_LIM, min(self.POS_LIM, tgt))  # hard clamp
            o   = self._make_order(p, tgt, cur, od)
            if o:
                result[p] = [o]

        # 4b. Close any orphaned position not owned by an active strategy.
        if self.ENABLE_CLOSE_UNTRACKED:
            for p, cur in state.position.items():
                if cur != 0 and p not in desired:
                    if p not in state.order_depths:
                        continue
                    od = state.order_depths[p]
                    o  = self._make_order(p, 0, cur, od)
                    if o:
                        result[p] = [o]

        # ── 5. Update and serialise state ─────────────────────────────── #

        # Decrement MR ticks; drop entries that have expired.
        new_mr = {p: t - 1 for p, t in mr_ticks.items() if t > 1}

        # Decrement snack signal ticks; drop expired entries.
        new_ss = {p: [d, t - 1] for p, (d, t) in snack_sig.items() if t > 1}

        # Persist only prices for products actually used by strategies.
        tracked = (
            set(self.TREND.keys())
            | {self.SNACK_LEADER}
            | set(self.SNACK_FOLLOWERS)
        )
        new_prev = {p: cur_mid[p] for p in tracked if p in cur_mid}

        trader_data = json.dumps({"p": new_prev, "m": new_mr, "s": new_ss})

        return result, 0, trader_data