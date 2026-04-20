from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import math


class Trader:

    LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    OSMIUM_FV = 10_000

    # ── Osmium parameters ────────────────────────────
    OSM_SWEEP_SELL_EDGE = 2     # unchanged
    OSM_INV_SWEEP_SCALE = 4     # unchanged
    OSM_BUY_EDGE        = 0     # ← CHANGED from 2
    OSM_SELL_EDGE        = 5    # ← CHANGED from 3
    OSM_SKEW            = 6     # unchanged
    OSM_PASSIVE_CAP     = 15    # unchanged

    # ── Pepper (ALL UNCHANGED) ───────────────────────
    PEPPER_SLOPE     = 0.0013
    HIST_LEN         = 50
    IMB_COEFF        = 0.3
    PEP_BIAS         = 2
    PEP_EDGE_BUY     = 1
    PEP_EDGE_SELL    = 5
    PEP_SKEW         = 3
    PEP_TARGET       = 70
    PEP_BUILD_AGGR   = 1
    PEP_SELL_THR     = 2
    PEP_HOLD_THR     = 4
    PEP_HOLD_SELL_QTY = 5

    def bid(self):
        return 15

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
                result[product] = self._trade_pepper(product, od, pos, lim, fv)

            else:
                result[product] = []

        return result, 0, json.dumps(data)

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

    def _pepper_fv(self, od, ts, data):
        mid = self._mid(od)
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
        if len(hist) < 8:
            return round(data["pi"] + self.PEPPER_SLOPE * (ts - data["pt"]))
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

    # ═══════════════════════════════════════════════
    #  OSMIUM — identical structure, two params changed
    # ═══════════════════════════════════════════════

    def _trade_osmium(self, od, pos, lim):
        S  = "ASH_COATED_OSMIUM"
        fv = self.OSMIUM_FV
        orders: List[Order] = []
        mb = max(0, lim - pos)
        ms = max(0, lim + pos)
        bu = su = 0

        ratio     = pos / lim if lim else 0.0
        abs_ratio = abs(ratio)

        short_pen = round(max(0, -ratio) * (1 + max(0, -ratio))
                         * self.OSM_INV_SWEEP_SCALE)
        long_pen  = round(max(0,  ratio) * (1 + max(0,  ratio))
                         * self.OSM_INV_SWEEP_SCALE)

        sell_sweep_min = fv + self.OSM_SWEEP_SELL_EDGE + short_pen
        buy_sweep_max  = fv - long_pen

        for px in sorted(od.sell_orders):
            if px <= buy_sweep_max and bu < mb:
                q = min(-od.sell_orders[px], mb - bu)
                if q > 0:
                    orders.append(Order(S, px, q)); bu += q

        for px in sorted(od.buy_orders, reverse=True):
            if px >= sell_sweep_min and su < ms:
                q = min(od.buy_orders[px], ms - su)
                if q > 0:
                    orders.append(Order(S, px, -q)); su += q

        ep = pos + bu - su
        if ep > 0 and fv in od.buy_orders and su < ms:
            q = min(od.buy_orders[fv], ms - su, ep)
            if q > 0:
                orders.append(Order(S, fv, -q)); su += q
        elif ep < 0 and fv in od.sell_orders and bu < mb:
            q = min(-od.sell_orders[fv], mb - bu, -ep)
            if q > 0:
                orders.append(Order(S, fv, q)); bu += q

        rb = max(0, mb - bu)
        rs = max(0, ms - su)

        sk = round(self.OSM_SKEW * ratio * (1 + abs_ratio))

        bpx_raw = fv - self.OSM_BUY_EDGE  - sk
        spx_raw = fv + self.OSM_SELL_EDGE - sk

        bpx = min(bpx_raw, fv)
        spx = max(spx_raw, fv)

        if pos < -25:
            rs = min(rs, self.OSM_PASSIVE_CAP)
        elif pos > 25:
            rb = min(rb, self.OSM_PASSIVE_CAP)

        if rb > 0:
            q1 = max(1, rb * 2 // 3)
            orders.append(Order(S, int(bpx), q1))
            if rb - q1 > 0:
                orders.append(Order(S, int(bpx - 1), rb - q1))

        if rs > 0:
            q1 = max(1, rs * 2 // 3)
            orders.append(Order(S, int(spx), -q1))
            if rs - q1 > 0:
                orders.append(Order(S, int(spx + 1), -(rs - q1)))

        return orders

    # ═══════════════════════════════════════════════
    #  PEPPER — completely unchanged
    # ═══════════════════════════════════════════════

    def _trade_pepper(self, sym, od, pos, lim, fv):
        S = sym
        orders: List[Order] = []
        mb = max(0, lim - pos)
        ms = max(0, lim + pos)
        bu = su = 0
        building = pos < self.PEP_TARGET

        bt = fv + (self.PEP_BUILD_AGGR if building else 0)
        for px in sorted(od.sell_orders):
            if px <= bt and bu < mb:
                q = min(-od.sell_orders[px], mb - bu)
                if q > 0:
                    orders.append(Order(S, px, q)); bu += q

        st = fv + (self.PEP_SELL_THR if building else self.PEP_HOLD_THR)
        for px in sorted(od.buy_orders, reverse=True):
            if px > st and su < ms:
                q = min(od.buy_orders[px], ms - su)
                if q > 0:
                    orders.append(Order(S, px, -q)); su += q

        rb = max(0, mb - bu)
        r   = pos / lim if lim else 0.0
        raw = self.PEP_SKEW * r * (1 + abs(r))
        if pos > 0:
            raw *= 0.3
        sk  = round(raw)
        bpx = min(fv - self.PEP_EDGE_BUY - sk, fv - 1)

        if rb > 0:
            orders.append(Order(S, bpx, rb))

        if not building:
            rs  = min(self.PEP_HOLD_SELL_QTY, max(0, ms - su))
            spx = max(fv + self.PEP_EDGE_SELL - sk, fv + 1)
            if rs > 0:
                orders.append(Order(S, spx, -rs))

        return orders