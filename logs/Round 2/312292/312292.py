from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class Trader:

    LIMITS = {
        "ASH_COATED_OSMIUM": 80,
        "INTARIAN_PEPPER_ROOT": 80,
    }

    # ── Osmium: static anchor ────────────────
    OSMIUM_FV = 10_000

    # ── Osmium: dynamic FV ───────────────────
    OSM_EMA_ALPHA = 0.01
    OSM_MR_WEIGHT = 0.3
    OSM_FV_CLAMP  = 3             # widened from 2 to give flow signals room

    # ── Osmium: order-flow params ────────────
    OSM_NPC_SHIFT_ALPHA = 0.15    # EMA smoothing on NPC mid deltas
    OSM_NPC_SHIFT_COEFF = 0.5     # 1-tick NPC shift → 0.5 FV adjustment
    OSM_TFLOW_ALPHA     = 0.3     # EMA smoothing on trade-flow
    OSM_TFLOW_COEFF     = 0.08    # per-lot aggressive flow → FV impact
    OSM_DEPLETION_COEFF = 0.5     # FV nudge when a whole book side is gone

    # ── Osmium: trading ──────────────────────
    OSM_SWEEP_SELL_EDGE = 2
    OSM_INV_SWEEP_SCALE = 3
    OSM_BUY_EDGE        = 0
    OSM_SELL_EDGE        = 4
    OSM_SKEW            = 4
    OSM_PASSIVE_CAP     = 20

    # ── Pepper ───────────────────────────────
    PEPPER_SLOPE      = 0.0013
    HIST_LEN          = 50
    IMB_COEFF         = 0.3
    PEP_BIAS          = 2
    PEP_EDGE_BUY      = 1
    PEP_EDGE_SELL     = 5
    PEP_SKEW          = 3
    PEP_TARGET        = 70
    PEP_BUILD_AGGR    = 2
    PEP_SELL_THR      = 3
    PEP_HOLD_THR      = 4
    PEP_HOLD_SELL_QTY = 5

    # ── Pepper: order-flow params ────────────
    PEP_NPC_SHIFT_ALPHA = 0.2     # faster alpha — captures trend acceleration
    PEP_NPC_SHIFT_COEFF = 0.3     # smaller coeff — trend already in base FV
    PEP_TFLOW_ALPHA     = 0.3
    PEP_TFLOW_COEFF     = 0.1

    # ─────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────

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
                flow_adj = self._osm_flow_adjustment(od, state, data)
                fv = self._osmium_dynamic_fv(od, data, flow_adj)
                result[product] = self._trade_osmium(od, pos, lim, fv)
                self._save_book_snapshot(od, data, "osm")

            elif product == "INTARIAN_PEPPER_ROOT":
                pep_adj = self._pep_flow_adjustment(od, state, data)
                base = self._pepper_fv(od, state.timestamp, data)
                if base is None:
                    result[product] = []
                    self._save_book_snapshot(od, data, "pep")
                    continue
                fv = round(base + self._microprice(od) + self.PEP_BIAS + pep_adj)
                result[product] = self._trade_pepper(product, od, pos, lim, fv)
                self._save_book_snapshot(od, data, "pep")

            else:
                result[product] = []

        return result, 0, json.dumps(data)

    # ─────────────────────────────────────────
    #  ORDER-FLOW SIGNAL ENGINE
    # ─────────────────────────────────────────

    def _npc_mid_shift(self, od, data, prefix, alpha, coeff):
        """
        Signal 1 — NPC mid-price shift.

        The NPC market-maker quotes symmetrically (inner bid vol ≈
        inner ask vol). When it moves its entire quote grid up or
        down, the mid-price shifts. An EMA of these deltas captures
        directional momentum from the NPC's own fair-value update.

        Returns a signed FV adjustment in price units.
        """
        ema_key  = f"{prefix}_shift_ema"

        if not od.buy_orders or not od.sell_orders:
            # Book one-sided — decay the EMA but don't update mid
            old = data.get(ema_key, 0.0)
            data[ema_key] = (1.0 - alpha) * old
            return data[ema_key] * coeff

        bb  = max(od.buy_orders)
        ba  = min(od.sell_orders)
        mid = (bb + ba) / 2.0

        prev_key = f"{prefix}_prev_npc_mid"

        if prev_key not in data:
            data[prev_key] = mid
            data[ema_key]  = 0.0
            return 0.0

        delta = mid - data[prev_key]
        data[prev_key] = mid

        old = data.get(ema_key, 0.0)
        data[ema_key] = alpha * delta + (1.0 - alpha) * old

        return data[ema_key] * coeff

    def _trade_flow(self, state, product, data, prefix, alpha, coeff):
        """
        Signal 2 — aggressive trade flow from market_trades.

        For each bot-to-bot trade, we compare its price to the
        PREVIOUS tick's best bid/ask to infer the aggressor side:
          trade price ≥ prev best ask  →  aggressive buy   (+qty)
          trade price ≤ prev best bid  →  aggressive sell  (−qty)

        An EMA of net signed lots is converted to a price adjustment.
        """
        ema_key = f"{prefix}_tflow"
        old_ema = data.get(ema_key, 0.0)

        prev_bb = data.get(f"{prefix}_snap_bb")
        prev_ba = data.get(f"{prefix}_snap_ba")

        net = 0
        for t in state.market_trades.get(product, []):
            # skip trades we participated in
            if t.buyer == "SUBMISSION" or t.seller == "SUBMISSION":
                continue
            if prev_ba is not None and t.price >= prev_ba:
                net += t.quantity          # aggressive buy
            elif prev_bb is not None and t.price <= prev_bb:
                net -= t.quantity          # aggressive sell

        data[ema_key] = alpha * net + (1.0 - alpha) * old_ema

        return data[ema_key] * coeff

    @staticmethod
    def _book_depletion(od, coeff):
        """
        Signal 3 — book-side depletion.

        The NPC normally quotes 2 levels on each side. When an
        entire side is empty, aggressive directional flow consumed
        it. This is a strong short-term momentum signal.

        When both sides are present, a milder volume-imbalance
        measure is used instead.
        """
        has_bids = bool(od.buy_orders)
        has_asks = bool(od.sell_orders)

        if not has_bids and has_asks:
            return -coeff          # bids consumed → bearish
        if has_bids and not has_asks:
            return  coeff          # asks consumed → bullish
        if not has_bids and not has_asks:
            return 0.0

        # both sides present — use volume imbalance as softer signal
        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(-v for v in od.sell_orders.values())
        total   = bid_vol + ask_vol
        if total == 0:
            return 0.0

        # more bids remaining → asks were consumed → bullish
        imbalance = (bid_vol - ask_vol) / total
        return imbalance * coeff * 0.3

    # ── composite adjustments ────────────────

    def _osm_flow_adjustment(self, od, state, data):
        s1 = self._npc_mid_shift(
            od, data, "osm",
            self.OSM_NPC_SHIFT_ALPHA, self.OSM_NPC_SHIFT_COEFF,
        )
        s2 = self._trade_flow(
            state, "ASH_COATED_OSMIUM", data, "osm",
            self.OSM_TFLOW_ALPHA, self.OSM_TFLOW_COEFF,
        )
        s3 = self._book_depletion(od, self.OSM_DEPLETION_COEFF)
        return s1 + s2 + s3

    def _pep_flow_adjustment(self, od, state, data):
        s1 = self._npc_mid_shift(
            od, data, "pep",
            self.PEP_NPC_SHIFT_ALPHA, self.PEP_NPC_SHIFT_COEFF,
        )
        s2 = self._trade_flow(
            state, "INTARIAN_PEPPER_ROOT", data, "pep",
            self.PEP_TFLOW_ALPHA, self.PEP_TFLOW_COEFF,
        )
        return s1 + s2

    def _save_book_snapshot(self, od, data, prefix):
        """Persist best-bid / best-ask for next tick's aggressor detection."""
        if od.buy_orders:
            data[f"{prefix}_snap_bb"] = max(od.buy_orders)
        else:
            data.pop(f"{prefix}_snap_bb", None)
        if od.sell_orders:
            data[f"{prefix}_snap_ba"] = min(od.sell_orders)
        else:
            data.pop(f"{prefix}_snap_ba", None)

    # ─────────────────────────────────────────
    #  OSMIUM DYNAMIC FV  (microprice EMA + anchor + flow)
    # ─────────────────────────────────────────

    def _osmium_dynamic_fv(self, od, data, flow_adj=0.0):
        static = self.OSMIUM_FV

        if not od.buy_orders or not od.sell_orders:
            return data.get("osm_fv", static)

        bb = max(od.buy_orders)
        ba = min(od.sell_orders)
        bv = od.buy_orders[bb]
        av = -od.sell_orders[ba]
        tot = bv + av

        micro = (ba * bv + bb * av) / tot if tot > 0 else (bb + ba) / 2.0

        if "osm_ema" not in data:
            data["osm_ema"] = float(micro)

        a = self.OSM_EMA_ALPHA
        data["osm_ema"] = a * micro + (1 - a) * data["osm_ema"]
        ema = data["osm_ema"]

        w   = self.OSM_MR_WEIGHT
        raw = w * static + (1 - w) * ema + flow_adj

        lo = static - self.OSM_FV_CLAMP
        hi = static + self.OSM_FV_CLAMP
        fv = round(max(lo, min(hi, raw)))

        data["osm_fv"] = fv
        return fv

    # ─────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────

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

    # ─────────────────────────────────────────
    #  OSMIUM TRADING
    # ─────────────────────────────────────────

    def _trade_osmium(self, od, pos, lim, fv):
        S = "ASH_COATED_OSMIUM"
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

        # ── sweeps ──
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

        # ── inventory mid-sell / mid-buy ──
        ep = pos + bu - su
        if ep > 0:
            for sell_px in [fv + 1, fv]:
                if sell_px in od.buy_orders and su < ms:
                    q = min(od.buy_orders[sell_px], ms - su,
                            max(0, pos + bu - su))
                    if q > 0:
                        orders.append(Order(S, sell_px, -q)); su += q
        elif ep < 0:
            for buy_px in [fv - 1, fv]:
                if buy_px in od.sell_orders and bu < mb:
                    q = min(-od.sell_orders[buy_px], mb - bu,
                            max(0, -(pos + bu - su)))
                    if q > 0:
                        orders.append(Order(S, buy_px, q)); bu += q

        # ── passive quotes ──
        rb = max(0, mb - bu)
        rs = max(0, ms - su)

        sk = round(self.OSM_SKEW * ratio * (1 + abs_ratio))

        bpx_raw = fv - self.OSM_BUY_EDGE - sk
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

    # ─────────────────────────────────────────
    #  PEPPER TRADING
    # ─────────────────────────────────────────

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