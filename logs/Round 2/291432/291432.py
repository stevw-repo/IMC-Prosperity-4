from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class Trader:

    LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # ─────── Osmium (stationary) ──────────────────────
    OSMIUM_FV = 10_000
    OSM_EDGE  = 2           # passive half-width
    OSM_SKEW  = 6           # inventory skew coefficient

    # ─────── Pepper (linear trend) ────────────────────
    PEPPER_SLOPE    = 0.0013  # price per timestamp unit
    PEP_BIAS        = 2       # ticks added to FV → shifts ALL order logic bullish
    PEP_EDGE_BUY    = 1       # tight passive buy (we WANT fills)
    PEP_EDGE_SELL   = 5       # wide passive sell when holding
    PEP_SKEW        = 3       # inventory skew (heavily damped when long)
    PEP_TARGET      = 70      # desired long position before we stop building
    PEP_BUILD_AGGR  = 1       # extra ticks willing to pay during build phase
    PEP_SELL_THR    = 2       # sweep-sell only if bid > fv + this (building)
    PEP_HOLD_THR    = 4       # sweep-sell only if bid > fv + this (holding)
    PEP_HOLD_SELL_QTY = 5     # max passive sell qty once at target

    # ─────── Shared ───────────────────────────────────
    HIST_LEN  = 50
    IMB_COEFF = 0.3           # microprice weighting

    def bid(self):
        return 15

    # ═══════════════════════════════════════════════════
    #  RUN
    # ═══════════════════════════════════════════════════

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        data: dict = {}
        if state.traderData:
            try:
                data = json.loads(state.traderData)
            except Exception:
                data = {}

        for product in state.order_depths:
            od  = state.order_depths[product]
            pos = state.position.get(product, 0)
            lim = self.LIMITS.get(product, 80)

            if product == "ASH_COATED_OSMIUM":
                result[product] = self._trade_osmium(od, pos, lim)

            elif product == "INTARIAN_PEPPER_ROOT":
                base = self._pepper_fv(od, state.timestamp, data)
                if base is None:
                    result[product] = []
                    continue
                fv = round(base + self._microprice(od) + self.PEP_BIAS)
                result[product] = self._trade_pepper(od, pos, lim, fv)

            else:
                result[product] = []

        return result, 0, json.dumps(data)

    # ═══════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════

    @staticmethod
    def _mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    def _microprice(self, od: OrderDepth) -> float:
        if not od.buy_orders or not od.sell_orders:
            return 0.0
        bb, ba = max(od.buy_orders), min(od.sell_orders)
        bv, av = od.buy_orders[bb], -od.sell_orders[ba]
        t = bv + av
        if t == 0:
            return 0.0
        return (bv - av) / t * (ba - bb) / 2.0 * self.IMB_COEFF

    # ── Pepper FV: dynamic intercept + online OLS ─────

    def _pepper_fv(self, od, ts, data):
        mid = self._mid(od)

        # First valid observation → set dynamic intercept
        if "pi" not in data:
            if mid and mid > 0:
                data["pi"] = mid
                data["pt"] = ts
            else:
                return None

        hist: list = data.get("ph", [])
        if mid and mid > 0:
            hist.append([ts, mid])
        if len(hist) > self.HIST_LEN:
            hist = hist[-self.HIST_LEN:]
        data["ph"] = hist

        # Bootstrap with calibrated slope until enough data
        if len(hist) < 8:
            return round(data["pi"] + self.PEPPER_SLOPE * (ts - data["pt"]))

        # Online OLS
        n  = len(hist)
        ta = [h[0] for h in hist]
        pa = [h[1] for h in hist]
        tb = sum(ta) / n
        pb = sum(pa) / n
        cov = sum((t - tb) * (p - pb) for t, p in zip(ta, pa))
        var = sum((t - tb) ** 2 for t in ta)
        if var > 0:
            return round(pb + (cov / var) * (ts - tb))
        return round(pa[-1])

    # ═══════════════════════════════════════════════════
    #  OSMIUM — symmetric market-making at known FV
    # ═══════════════════════════════════════════════════

    def _trade_osmium(self, od, pos, lim):
        S  = "ASH_COATED_OSMIUM"
        fv = self.OSMIUM_FV                     # known with certainty
        orders: List[Order] = []
        mb = max(0, lim - pos)                  # max buy qty
        ms = max(0, lim + pos)                  # max sell qty
        bu = su = 0

        # ── L1: sweep all mispriced levels ──
        for px in sorted(od.sell_orders):
            if px < fv and bu < mb:
                q = min(-od.sell_orders[px], mb - bu)
                if q > 0:
                    orders.append(Order(S, px, q)); bu += q

        for px in sorted(od.buy_orders, reverse=True):
            if px > fv and su < ms:
                q = min(od.buy_orders[px], ms - su)
                if q > 0:
                    orders.append(Order(S, px, -q)); su += q

        # ── L1b: take at exactly FV to flatten inventory ──
        ep = pos + bu - su
        if ep > 0 and fv in od.buy_orders:
            q = min(od.buy_orders[fv], ms - su, ep)
            if q > 0:
                orders.append(Order(S, fv, -q)); su += q
        elif ep < 0 and fv in od.sell_orders:
            q = min(-od.sell_orders[fv], mb - bu, -ep)
            if q > 0:
                orders.append(Order(S, fv, q)); bu += q

        # ── L2: passive quotes with inventory skew ──
        rb = max(0, mb - bu)
        rs = max(0, ms - su)

        r  = pos / lim if lim else 0.0
        sk = round(self.OSM_SKEW * r * (1 + abs(r)))

        bpx = min(fv - self.OSM_EDGE - sk, fv - 1)
        spx = max(fv + self.OSM_EDGE - sk, fv + 1)

        # split across two levels for depth
        if rb > 0:
            q1 = max(1, rb * 2 // 3)
            orders.append(Order(S, bpx, q1))
            if rb - q1 > 0:
                orders.append(Order(S, bpx - 1, rb - q1))
        if rs > 0:
            q1 = max(1, rs * 2 // 3)
            orders.append(Order(S, spx, -q1))
            if rs - q1 > 0:
                orders.append(Order(S, spx + 1, -(rs - q1)))

        return orders

    # ═══════════════════════════════════════════════════
    #  PEPPER — trend-loading market-making
    # ═══════════════════════════════════════════════════

    def _trade_pepper(self, od, pos, lim, fv):
        S = "INTARIAN_PEPPER_ROOT"
        orders: List[Order] = []
        mb = max(0, lim - pos)
        ms = max(0, lim + pos)
        bu = su = 0

        building = pos < self.PEP_TARGET

        # ── L1: aggressive buy sweep ──
        #   fv already includes +2 bias, so this buys
        #   up to ~mid+3 when building
        bt = fv + (self.PEP_BUILD_AGGR if building else 0)
        for px in sorted(od.sell_orders):
            if px <= bt and bu < mb:
                q = min(-od.sell_orders[px], mb - bu)
                if q > 0:
                    orders.append(Order(S, px, q)); bu += q

        # ── L1: very reluctant sell sweep ──
        #   building: only sell bids > ~mid+4
        #   holding:  only sell bids > ~mid+6
        st = fv + (self.PEP_SELL_THR if building else self.PEP_HOLD_THR)
        for px in sorted(od.buy_orders, reverse=True):
            if px > st and su < ms:
                q = min(od.buy_orders[px], ms - su)
                if q > 0:
                    orders.append(Order(S, px, -q)); su += q

        # ── L2: passive quotes ──
        rb = max(0, mb - bu)

        # skew: heavily damped when long (we WANT to be long)
        r   = pos / lim if lim else 0.0
        raw = self.PEP_SKEW * r * (1 + abs(r))
        if pos > 0:
            raw *= 0.3
        sk = round(raw)

        bpx = min(fv - self.PEP_EDGE_BUY - sk, fv - 1)

        if rb > 0:
            orders.append(Order(S, bpx, rb))

        # passive sells: NONE during building; tiny when holding
        if not building:
            rs  = min(self.PEP_HOLD_SELL_QTY, max(0, ms - su))
            spx = max(fv + self.PEP_EDGE_SELL - sk, fv + 1)
            if rs > 0:
                orders.append(Order(S, spx, -rs))

        return orders