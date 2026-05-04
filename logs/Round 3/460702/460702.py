from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import math
import json


class Trader:

    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
        "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
        "VEV_5400": 300, "VEV_5500": 300, "VEV_6000": 300,
        "VEV_6500": 300,
    }

    VOUCHER_STRIKES = {
        "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
        "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
        "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    # Strikes we actively BUY (near-ATM, large mispricing)
    ACTIVE_OPTION_STRIKES = [
        "VEV_5000", "VEV_5100", "VEV_5200",
        "VEV_5300", "VEV_5400", "VEV_5500",
    ]

    REALIZED_VOL = 0.0215          # daily vol from 3 days of data
    BASE_TTE = 5.0                 # TTE at start of round 3
    TICKS_PER_DAY = 1_000_000      # timestamp range per day

    # ── Black-Scholes helpers (r = 0) ──────────────────────────

    @staticmethod
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def norm_pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def bs_price(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 1e-10 or sigma <= 1e-10:
            return max(S - K, 0.0)
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * self.norm_cdf(d1) - K * self.norm_cdf(d2)

    def bs_delta(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 1e-10 or sigma <= 1e-10:
            return 1.0 if S > K else 0.0
        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
        return self.norm_cdf(d1)

    # ── Order-book helpers ─────────────────────────────────────

    @staticmethod
    def get_mid(od: OrderDepth):
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    @staticmethod
    def get_microprice(od: OrderDepth):
        """Volume-weighted mid — better fair-value estimator."""
        if od.buy_orders and od.sell_orders:
            bb = max(od.buy_orders)
            ba = min(od.sell_orders)
            bv = od.buy_orders[bb]
            av = -od.sell_orders[ba]          # make positive
            if bv + av > 0:
                return (bb * av + ba * bv) / (bv + av)
            return (bb + ba) / 2.0
        return None

    # ── Main entry point ───────────────────────────────────────

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Time to expiry (linearly decays over the day)
        TTE = max(self.BASE_TTE - state.timestamp / self.TICKS_PER_DAY, 0.001)

        # Underlying mid price
        ve_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            ve_mid = self.get_mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        # ── 1) HYDROGEL_PACK  market making ────────────────────
        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self._trade_hydrogel(state)

        # ── 2) OPTIONS  buy underpriced near-ATM ───────────────
        if ve_mid is not None and ve_mid > 0:
            for sym in self.ACTIVE_OPTION_STRIKES:
                if sym in state.order_depths:
                    K = self.VOUCHER_STRIKES[sym]
                    result[sym] = self._trade_option(
                        state, sym, K, ve_mid, TTE
                    )

        # ── 3) Compute net option delta for hedging ────────────
        net_delta = 0.0
        if ve_mid is not None and ve_mid > 0:
            for sym, K in self.VOUCHER_STRIKES.items():
                pos = state.position.get(sym, 0)
                if pos != 0:
                    net_delta += pos * self.bs_delta(
                        ve_mid, K, TTE, self.REALIZED_VOL
                    )

        # ── 4) VELVETFRUIT_EXTRACT  delta hedge ────────────────
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self._trade_ve_hedge(
                state, net_delta
            )

        traderData = ""
        return result, conversions, traderData

    # ── Strategy: HYDROGEL_PACK market making ──────────────────

    def _trade_hydrogel(self, state: TradingState) -> List[Order]:
        product = "HYDROGEL_PACK"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        fair = self.get_microprice(od)
        if fair is None:
            return orders

        # Inventory-adjusted fair value (push fair toward flat position)
        adj_fair = fair - 0.08 * pos

        buy_budget = limit - pos        # max we can buy
        sell_budget = limit + pos       # max we can sell

        # ── Sweep mispriced asks ──
        for price in sorted(od.sell_orders):
            if buy_budget <= 0:
                break
            vol = -od.sell_orders[price]
            if price < adj_fair - 0.5:
                qty = min(vol, buy_budget)
                orders.append(Order(product, price, qty))
                buy_budget -= qty

        # ── Sweep mispriced bids ──
        for price in sorted(od.buy_orders, reverse=True):
            if sell_budget <= 0:
                break
            vol = od.buy_orders[price]
            if price > adj_fair + 0.5:
                qty = min(vol, sell_budget)
                orders.append(Order(product, price, -qty))
                sell_budget -= qty

        # ── Post resting quotes inside the spread ──
        bid_px = int(round(adj_fair)) - 2
        ask_px = int(round(adj_fair)) + 2

        if buy_budget > 0:
            orders.append(Order(product, bid_px, buy_budget))
        if sell_budget > 0:
            orders.append(Order(product, ask_px, -sell_budget))

        return orders

    # ── Strategy: buy underpriced options ───────────────────────

    def _trade_option(
        self, state: TradingState,
        sym: str, K: int, S: float, TTE: float
    ) -> List[Order]:
        od = state.order_depths[sym]
        pos = state.position.get(sym, 0)
        limit = self.POSITION_LIMITS[sym]
        orders: List[Order] = []

        theo = self.bs_price(S, K, TTE, self.REALIZED_VOL)
        delta = self.bs_delta(S, K, TTE, self.REALIZED_VOL)

        # Skip options that are too deep ITM or OTM
        if delta > 0.95 or delta < 0.02:
            return orders

        buy_budget = limit - pos
        sell_budget = limit + pos

        # ── Aggressively buy everything offered below theo ──
        for price in sorted(od.sell_orders):
            if buy_budget <= 0:
                break
            vol = -od.sell_orders[price]
            if price < theo:
                qty = min(vol, buy_budget)
                orders.append(Order(sym, price, qty))
                buy_budget -= qty

        # ── Post a resting bid to accumulate slowly ──
        if buy_budget > 0 and od.buy_orders:
            best_bid = max(od.buy_orders)
            # Bid just above best market bid, but stay below theo
            my_bid = min(int(round(theo)) - 1, best_bid + 1)
            my_bid = max(my_bid, 1)
            if my_bid < theo:
                orders.append(Order(sym, my_bid, buy_budget))

        # ── Sell if somehow overpriced vs realized vol (rare) ──
        for price in sorted(od.buy_orders, reverse=True):
            if sell_budget <= 0:
                break
            vol = od.buy_orders[price]
            if price > theo + 1:
                qty = min(vol, sell_budget)
                orders.append(Order(sym, price, -qty))
                sell_budget -= qty

        return orders

    # ── Strategy: VELVETFRUIT_EXTRACT delta hedge ──────────────

    def _trade_ve_hedge(
        self, state: TradingState, net_option_delta: float
    ) -> List[Order]:
        product = "VELVETFRUIT_EXTRACT"
        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        orders: List[Order] = []

        mid = self.get_mid(od)
        if mid is None:
            return orders

        # Target: offset the option delta (clamped to position limits)
        target = int(round(-net_option_delta))
        target = max(-limit, min(limit, target))

        diff = target - pos                       # how much we need to trade
        buy_budget = limit - pos
        sell_budget = limit + pos

        if diff > 0:
            # ── Need to BUY underlying ──
            qty_left = min(diff, buy_budget)
            for price in sorted(od.sell_orders):
                if qty_left <= 0:
                    break
                vol = -od.sell_orders[price]
                qty = min(vol, qty_left)
                orders.append(Order(product, price, qty))
                qty_left -= qty
            if qty_left > 0:
                bb = max(od.buy_orders) if od.buy_orders else int(mid) - 3
                orders.append(Order(product, bb + 1, qty_left))

        elif diff < 0:
            # ── Need to SELL underlying ──
            qty_left = min(-diff, sell_budget)
            for price in sorted(od.buy_orders, reverse=True):
                if qty_left <= 0:
                    break
                vol = od.buy_orders[price]
                qty = min(vol, qty_left)
                orders.append(Order(product, price, -qty))
                qty_left -= qty
            if qty_left > 0:
                ba = min(od.sell_orders) if od.sell_orders else int(mid) + 3
                orders.append(Order(product, ba - 1, -qty_left))

        else:
            # ── At target: market-make for extra income ──
            mm = min(20, buy_budget, sell_budget)
            if mm > 0:
                orders.append(Order(product, int(mid) - 2, mm))
                orders.append(Order(product, int(mid) + 2, -mm))

        return orders