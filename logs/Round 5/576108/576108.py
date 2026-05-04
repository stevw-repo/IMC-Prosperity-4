from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple
import json
import math


class Trader:
    """
    IMC Prosperity 4 – Round 5  (v2 – corrected + extended)
    =========================================================

    Five independently togglable sub-strategies:

      1. DIRECTIONAL_TREND  – Hold ±POS_LIM (or ±PAIRS_MAX_POS for pairs
                              products) in the identified trend direction.
      2. MR_ROBOT_DISHES    – Adaptive-threshold contrarian short after a
                              large UP move on ROBOT_DISHES.
      3. PAIRS_ROBOT        – Stat-arb overlay on ROBOT_DISHES /
                              ROBOT_IRONING z-score.
      4. INTRADAY_TIMING    – H1 / H2 directional overrides.
      5. CLOSE_UNTRACKED    – Safety-net position flattening.

    CHANGELOG vs v1
    ---------------
    BUG FIX – SNACK_LEAD_LAG removed entirely.
        The strategy was forcing SNACKPACK_PISTACHIO (trend SHORT = -10)
        to +10 whenever SNACKPACK_RASPBERRY ticked up, and SNACKPACK_
        STRAWBERRY (trend LONG = +10) to -10 on RASPBERRY down moves.
        The resulting 20-unit round trips at SNACK_HOLD=3 ticks fired
        ~3,300 times per day at ~$300 spread cost each = -$2.3M observed
        loss.  PISTACHIO and STRAWBERRY are fully handled by
        DIRECTIONAL_TREND and need no lead-lag overlay.

    BUG FIX – UV_VISOR_ORANGE added to H1_ONLY.
        The 3-day backtest chart shows a clear inverted-U shape with a
        peak near the midpoint and a reversion to near-starting prices
        by end-of-day.  The long is now held only during H1.

    NEW – Adaptive MR threshold.
        THRESHOLD = max(MR_K × rolling_std(last MR_VOL_WINDOW returns),
                        MR_FLOOR).
        Computed from the PREVIOUS window before the current tick is
        appended, avoiding circularity.  Eliminates fixed-threshold
        overfitting to a single volatility regime.

    NEW – PAIRS_ROBOT strategy.
        Detrended spread = DISHES − PAIRS_BETA × IRONING with z-score
        entry/exit.  The trend base for both legs is reduced to
        PAIRS_MAX_POS so the ±PAIRS_MAX_POS overlay can reach ±POS_LIM
        in either direction without clamping.

    ⚠  PAIRS_BETA: verify against your OLS regression.  Theoretical
       value from sample slopes is −0.72 (negative because the two
       products move in opposite directions).  Update before going live.
    """

    # ------------------------------------------------------------------ #
    #  STRATEGY ENABLE / DISABLE                                           #
    # ------------------------------------------------------------------ #
    ENABLE_DIRECTIONAL_TREND  = True
    ENABLE_MR_ROBOT_DISHES    = True
    ENABLE_PAIRS_ROBOT        = True
    ENABLE_INTRADAY_TIMING    = True
    ENABLE_CLOSE_UNTRACKED    = True

    # ------------------------------------------------------------------ #
    #  EXCHANGE PARAMETERS                                                 #
    # ------------------------------------------------------------------ #
    POS_LIM = 10
    H1_END  = 500_000

    # ------------------------------------------------------------------ #
    #  DIRECTIONAL TREND TABLE  –  product -> (direction, R²)             #
    # ------------------------------------------------------------------ #
    TREND: Dict[str, Tuple[int, float]] = {
        # ── SHORTS ───────────────────────────────────────────────────── #
        "MICROCHIP_OVAL":               (-1, 0.912),
        "PEBBLES_XS":                   (-1, 0.900),
        "UV_VISOR_AMBER":               (-1, 0.912),
        "PEBBLES_S":                    (-1, 0.798),
        "ROBOT_IRONING":                (-1, 0.766),   # PAIRS_B
        "ROBOT_VACUUMING":              (-1, 0.764),
        "MICROCHIP_TRIANGLE":           (-1, 0.643),
        "ROBOT_LAUNDRY":                (-1, 0.629),
        "SNACKPACK_PISTACHIO":          (-1, 0.630),
        "MICROCHIP_RECTANGLE":          (-1, 0.571),
        "OXYGEN_SHAKE_MORNING_BREATH":  (-1, 0.556),
        "PANEL_1X4":                    (-1, 0.531),
        "TRANSLATOR_ASTRO_BLACK":       (-1, 0.477),
        "SNACKPACK_CHOCOLATE":          (-1, 0.452),
        "TRANSLATOR_SPACE_GRAY":        (-1, 0.398),
        "PANEL_2X2":                    (-1, 0.360),
        # ── LONGS ────────────────────────────────────────────────────── #
        "OXYGEN_SHAKE_GARLIC":          (+1, 0.806),
        "SLEEP_POD_POLYESTER":          (+1, 0.802),
        "GALAXY_SOUNDS_BLACK_HOLES":    (+1, 0.785),
        "PANEL_2X4":                    (+1, 0.782),
        "ROBOT_DISHES":                 (+1, 0.765),   # MR + PAIRS_A
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
        "UV_VISOR_ORANGE":              (+1, 0.363),   # H1_ONLY (see below)
        "OXYGEN_SHAKE_CHOCOLATE":       (+1, 0.354),
        "SLEEP_POD_NYLON":              (+1, 0.332),
    }

    # ------------------------------------------------------------------ #
    #  INTRADAY TIMING OVERRIDES                                           #
    # ------------------------------------------------------------------ #
    # H2_FLIP  : reverse direction in H2
    # H1_ONLY  : go flat in H2
    # H2_ONLY  : go flat in H1 (enter only after H1_END)
    H2_FLIP: set = {"SLEEP_POD_COTTON"}
    H1_ONLY: set = {
        "MICROCHIP_SQUARE",   # ~zero H2 slope in sample
        "UV_VISOR_ORANGE",    # H2 reversal confirmed in 3-day backtest
    }
    H2_ONLY: set = {
        "ROBOT_DISHES",               # H1 slope weak; H2 slope dominates
        "UV_VISOR_MAGENTA",           # same pattern
        "GALAXY_SOUNDS_BLACK_HOLES",  # back-loaded
    }

    # ------------------------------------------------------------------ #
    #  MEAN-REVERSION  –  ROBOT_DISHES                                    #
    # ------------------------------------------------------------------ #
    MR_PRODUCT    = "ROBOT_DISHES"
    MR_K          = 2.0    # adaptive threshold = MR_K × rolling_vol
    MR_FLOOR      = 5.0    # hard minimum threshold (price ticks)
    MR_VOL_WINDOW = 50     # look-back ticks for vol estimate
    MR_HOLD       = 30     # ticks to hold the contrarian short

    # ------------------------------------------------------------------ #
    #  PAIRS TRADE  –  ROBOT_DISHES (A) vs ROBOT_IRONING (B)              #
    #                                                                      #
    #  Spread definition:                                                  #
    #    spread(t) = price_A(t)  −  PAIRS_BETA × price_B(t)              #
    #                                                                      #
    #  Theoretical beta from sample slopes:                               #
    #    beta = −slope_A / slope_B = −(+0.056) / (−0.078) ≈ −0.72       #
    #  (negative because the two products trend in opposite directions)   #
    #                                                                      #
    #  Verify PAIRS_BETA against your OLS regression before going live.   #
    #                                                                      #
    #  Position sizing:                                                    #
    #    DIRECTIONAL_TREND uses PAIRS_MAX_POS as base for both legs so    #
    #    the ±PAIRS_MAX_POS overlay can swing the combined position all   #
    #    the way from 0 to ±POS_LIM without being clamped.               #
    #                                                                      #
    #    z > +ENTRY_Z (spread wide, A expensive):                         #
    #      trend_A = +5  →  +5 + (−5) = 0  (reduce long DISHES)         #
    #      trend_B = −5  →  −5 + (+5) = 0  (reduce short IRONING)       #
    #                                                                      #
    #    z < −ENTRY_Z (spread narrow, A cheap):                           #
    #      trend_A = +5  →  +5 + (+5) = +10  (max long DISHES)           #
    #      trend_B = −5  →  −5 + (−5) = −10  (max short IRONING)         #
    # ------------------------------------------------------------------ #
    PAIRS_A       = "ROBOT_DISHES"
    PAIRS_B       = "ROBOT_IRONING"
    PAIRS_BETA    = -0.72   # ⚠ verify from OLS regression
    PAIRS_WINDOW  = 200     # rolling window for spread mean / std
    PAIRS_WARMUP  = 150     # minimum observations before entering
    PAIRS_ENTRY_Z = 2.0     # |z| > entry → open
    PAIRS_EXIT_Z  = 0.5     # |z| < exit  → close
    PAIRS_MAX_POS = 5       # per-leg position magnitude

    # ================================================================== #
    #  PRIVATE HELPERS                                                     #
    # ================================================================== #

    @staticmethod
    def _best_bid(od: OrderDepth) -> Optional[int]:
        return max(od.buy_orders) if od.buy_orders else None

    @staticmethod
    def _best_ask(od: OrderDepth) -> Optional[int]:
        return min(od.sell_orders) if od.sell_orders else None

    @staticmethod
    def _mid(od: OrderDepth) -> Optional[float]:
        bb = max(od.buy_orders)  if od.buy_orders  else None
        ba = min(od.sell_orders) if od.sell_orders else None
        if bb is not None and ba is not None:
            return (bb + ba) / 2.0
        return float(bb) if bb is not None else (float(ba) if ba is not None else None)

    @staticmethod
    def _rolling_std(values: list) -> float:
        """Sample std; returns 0.0 for fewer than 2 observations."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))

    def _make_order(
        self,
        product:    str,
        target_pos: int,
        cur_pos:    int,
        od:         OrderDepth,
    ) -> Optional[Order]:
        """
        Single aggressive (liquidity-taking) order moving cur_pos toward
        target_pos.  Quantity is clamped so the resulting position never
        exceeds ±POS_LIM.  Returns None if no move is needed or the book
        is empty on the required side.
        """
        delta = target_pos - cur_pos
        if delta == 0:
            return None
        if delta > 0:                                   # need to BUY
            ba = self._best_ask(od)
            if ba is None:
                return None
            qty = min(delta, self.POS_LIM - cur_pos)
            return Order(product, ba, qty) if qty > 0 else None
        else:                                           # need to SELL
            bb = self._best_bid(od)
            if bb is None:
                return None
            qty = min(-delta, cur_pos + self.POS_LIM)
            return Order(product, bb, -qty) if qty > 0 else None

    # ================================================================== #
    #  MAIN ENTRY POINT                                                    #
    # ================================================================== #

    def run(self, state: TradingState):

        # ── 1. Deserialise persistent state ──────────────────────────── #
        try:
            sd = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            sd = {}

        # "p"  : previous mid prices         {product: float}
        # "m"  : MR ticks remaining          {product: int}
        # "mr" : recent ΔP for vol estimate  [float, ...]
        # "sh" : pairs spread history        [float, ...]
        # "ps" : pairs state                 -1 | 0 | +1
        prev_mid    : Dict[str, float] = sd.get("p",  {})
        mr_ticks    : Dict[str, int]   = sd.get("m",  {})
        mr_returns  : list             = sd.get("mr", [])
        spread_hist : list             = sd.get("sh", [])
        pairs_state : int              = sd.get("ps", 0)

        ts    = state.timestamp
        is_h2 = ts >= self.H1_END

        # ── 2. Compute current mid prices ────────────────────────────── #
        cur_mid: Dict[str, float] = {}
        for p, od in state.order_depths.items():
            m = self._mid(od)
            if m is not None:
                cur_mid[p] = m

        # ── 3. Adaptive MR threshold (uses PREVIOUS window) ───────────── #
        mr_p         = self.MR_PRODUCT
        mr_vol       = self._rolling_std(mr_returns)          # from previous ticks
        mr_threshold = max(self.MR_K * mr_vol, self.MR_FLOOR) # before appending current

        # Now append the current return
        if mr_p in cur_mid and mr_p in prev_mid:
            mr_returns.append(cur_mid[mr_p] - prev_mid[mr_p])
        mr_returns = mr_returns[-self.MR_VOL_WINDOW:]          # rolling trim

        # ── 4. Update pairs spread history ───────────────────────────── #
        pa, pb        = self.PAIRS_A, self.PAIRS_B
        raw_spread    : Optional[float] = None
        if pa in cur_mid and pb in cur_mid:
            raw_spread = cur_mid[pa] - self.PAIRS_BETA * cur_mid[pb]
            spread_hist.append(raw_spread)
        spread_hist = spread_hist[-self.PAIRS_WINDOW:]

        # Compute z-score if warmed up
        pairs_z : Optional[float] = None
        n_sh    = len(spread_hist)
        if raw_spread is not None and n_sh >= self.PAIRS_WARMUP:
            s_mean  = sum(spread_hist) / n_sh
            s_std   = self._rolling_std(spread_hist)
            if s_std > 0:
                pairs_z = (raw_spread - s_mean) / s_std

        # Pairs state machine: entry / exit
        if pairs_z is not None:
            if pairs_state == 0:                            # flat: check for entry
                if pairs_z > self.PAIRS_ENTRY_Z:
                    pairs_state = -1                        # short spread
                elif pairs_z < -self.PAIRS_ENTRY_Z:
                    pairs_state = +1                        # long spread
            else:                                          # in position: check exit
                if abs(pairs_z) < self.PAIRS_EXIT_Z:
                    pairs_state = 0                         # spread reverted → exit

        # ── 5. Build desired positions ────────────────────────────────── #
        desired: Dict[str, int] = {}

        # ── 5a. Directional trend ─────────────────────────────────────── #
        if self.ENABLE_DIRECTIONAL_TREND:
            for p, (direction, _) in self.TREND.items():
                if p not in state.order_depths:
                    continue

                eff = direction

                if self.ENABLE_INTRADAY_TIMING:
                    if is_h2:
                        if p in self.H2_FLIP:
                            eff = -direction    # reverse in H2
                        elif p in self.H1_ONLY:
                            eff = 0             # trend exhausted; go flat
                    else:                       # H1
                        if p in self.H2_ONLY:
                            eff = 0             # wait for H2 entry window

                # Pairs legs use a reduced base to leave room for the overlay
                if self.ENABLE_PAIRS_ROBOT and p in (pa, pb):
                    desired[p] = eff * self.PAIRS_MAX_POS
                else:
                    desired[p] = eff * self.POS_LIM

        # ── 5b. Pairs overlay ─────────────────────────────────────────── #
        if self.ENABLE_PAIRS_ROBOT and pairs_state != 0:
            for p in (pa, pb):
                if p not in state.order_depths:
                    continue

                if pairs_state == -1:           # short spread: short A, long B
                    delta = -self.PAIRS_MAX_POS if p == pa else +self.PAIRS_MAX_POS
                else:                           # long spread: long A, short B
                    delta = +self.PAIRS_MAX_POS if p == pa else -self.PAIRS_MAX_POS

                base      = desired.get(p, 0)
                desired[p] = max(-self.POS_LIM, min(self.POS_LIM, base + delta))

        # ── 5c. MR override – ROBOT_DISHES ───────────────────────────── #
        if self.ENABLE_MR_ROBOT_DISHES:
            if (mr_p in cur_mid and mr_p in prev_mid
                    and len(mr_returns) >= 5):
                dp = cur_mid[mr_p] - prev_mid[mr_p]
                if dp >= mr_threshold:
                    # Large UP move: reset contrarian short timer
                    mr_ticks[mr_p] = self.MR_HOLD

            if mr_p in mr_ticks:
                # Hard override: contrarian short takes priority over
                # both the directional trend and the pairs overlay
                desired[mr_p] = -self.POS_LIM

        # ── 6. Generate orders ────────────────────────────────────────── #
        result: Dict[str, List[Order]] = {}

        for p, tgt in desired.items():
            if p not in state.order_depths:
                continue
            od  = state.order_depths[p]
            cur = state.position.get(p, 0)
            tgt = max(-self.POS_LIM, min(self.POS_LIM, tgt))  # hard clamp
            o   = self._make_order(p, tgt, cur, od)
            if o:
                result[p] = [o]

        # Safety net: flatten any position not owned by an active strategy
        if self.ENABLE_CLOSE_UNTRACKED:
            for p, cur in state.position.items():
                if cur != 0 and p not in desired and p in state.order_depths:
                    od = state.order_depths[p]
                    o  = self._make_order(p, 0, cur, od)
                    if o:
                        result[p] = [o]

        # ── 7. Serialise state ────────────────────────────────────────── #
        tracked  = set(self.TREND.keys()) | {mr_p, pa, pb}
        new_prev = {p: cur_mid[p] for p in tracked if p in cur_mid}
        new_mr_t = {p: t - 1 for p, t in mr_ticks.items() if t > 1}

        trader_data = json.dumps({
            "p":  new_prev,
            "m":  new_mr_t,
            "mr": mr_returns,
            "sh": spread_hist,
            "ps": pairs_state,
        })

        return result, 0, trader_data