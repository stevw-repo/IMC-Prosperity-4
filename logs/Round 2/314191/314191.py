from datamodel import OrderDepth, TradingState, Order
from typing import List
import json
import math


class Trader:
    """
    Round 2 trading algorithm for ASH_COATED_OSMIUM and INTARIAN_PEPPER_ROOT.
    
    Strategy (adapted from Frankfurt Hedgehogs' Prosperity 3 approach):
    
    Both products share the same core logic:
      1. Estimate fair price via "Wall Mid" — average of deepest bid and ask levels
      2. TAKING: Aggressively sweep any orders mispriced vs. fair (extra tight levels)
      3. MAKING: Post passive orders at improved prices (overbid bids, undercut asks)
      4. INVENTORY: Flatten excess position at fair to free up risk capacity
    
    Product-specific adjustments:
      - Osmium: mean-reverts around 10,000 → no directional bias needed
      - Pepper Root: linear uptrend of ~0.001/timestamp → slight long bias via fair shift
    """

    LIMITS = {
        'ASH_COATED_OSMIUM': 80,
        'INTARIAN_PEPPER_ROOT': 80,
    }

    # Pepper Root trends up at ~0.1 per tick (0.001 per timestamp).
    # Shifting fair up by 1 creates a structural long bias:
    # - We buy 1 tick more aggressively
    # - We sell 1 tick less aggressively
    # - We hold long positions longer before flattening
    PEPPER_TREND_BIAS = 1

    def bid(self):
        """Market Access Fee bid for Round 2. Placeholder — set to 0 for now."""
        return 0

    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # ---- Deserialize persisted trader data ----
        trader_data = {}
        if state.traderData and state.traderData.strip():
            try:
                trader_data = json.loads(state.traderData)
            except:
                trader_data = {}

        new_trader_data = {}

        # ---- Trade each product ----
        for product in state.order_depths:
            if product in self.LIMITS:
                result[product] = self.trade(state, product, trader_data, new_trader_data)

        # ---- Serialize state for next iteration ----
        try:
            td_str = json.dumps(new_trader_data)
        except:
            td_str = ''

        return result, conversions, td_str

    # =========================================================================
    # CORE TRADING LOGIC
    # =========================================================================

    def trade(self, state: TradingState, product: str,
              td: dict, new_td: dict) -> List[Order]:

        orders: List[Order] = []
        position = state.position.get(product, 0)
        limit = self.LIMITS[product]

        # ----- Parse order book -----
        od = state.order_depths[product]

        # Bids from bots: {price: +volume}, sorted highest price first
        buys = {}
        if od.buy_orders:
            buys = {p: abs(v) for p, v in
                    sorted(od.buy_orders.items(), key=lambda x: -x[0])}

        # Asks from bots: {price: +volume}, sorted lowest price first
        sells = {}
        if od.sell_orders:
            sells = {p: abs(v) for p, v in
                     sorted(od.sell_orders.items(), key=lambda x: x[0])}

        if not buys and not sells:
            return orders

        # ----- Identify walls (deepest liquidity levels) -----
        bid_wall = min(buys.keys()) if buys else None
        ask_wall = max(sells.keys()) if sells else None
        best_bid = max(buys.keys()) if buys else None
        best_ask = min(sells.keys()) if sells else None

        # ----- Compute Wall Mid (fair price estimate) -----
        # The deepest levels are posted by market-maker bots that know the
        # true price and quote symmetrically around it.
        wall_mid = None
        if bid_wall is not None and ask_wall is not None:
            wall_mid = (bid_wall + ask_wall) / 2

        # Fallback: use last known wall mid if current book is one-sided
        if wall_mid is None:
            wall_mid = td.get(f'{product}_wm')

        if wall_mid is None:
            return orders  # Cannot trade without a fair price estimate

        # Persist wall mid for next tick's fallback
        new_td[f'{product}_wm'] = wall_mid

        # ----- Fair price with product-specific adjustment -----
        fair = wall_mid
        if product == 'INTARIAN_PEPPER_ROOT':
            fair += self.PEPPER_TREND_BIAS

        # ----- Remaining order capacity (respecting position limits) -----
        max_buy = limit - position
        max_sell = limit + position

        # =================================================================
        # PHASE 1 — TAKING (aggressive, immediate execution)
        #
        # Sweep any orders that are clearly mispriced relative to fair.
        # These are the "extra tight levels" that appear occasionally.
        # Also flatten inventory at fair price to manage risk.
        # =================================================================

        # Buy any asks priced at fair-1 or below (at least 1 tick of edge)
        for ap in sorted(sells.keys()):
            if max_buy <= 0:
                break
            if ap <= fair - 1:
                qty = min(sells[ap], max_buy)
                orders.append(Order(product, ap, qty))
                max_buy -= qty
            elif ap <= fair and position < 0:
                # Flatten short position at fair (zero edge, but reduces risk)
                qty = min(sells[ap], max_buy, abs(position))
                if qty > 0:
                    orders.append(Order(product, ap, qty))
                    max_buy -= qty

        # Sell to any bids priced at fair+1 or above
        for bp in sorted(buys.keys(), reverse=True):
            if max_sell <= 0:
                break
            if bp >= fair + 1:
                qty = min(buys[bp], max_sell)
                orders.append(Order(product, bp, -qty))
                max_sell -= qty
            elif bp >= fair and position > 0:
                # Flatten long position at fair
                qty = min(buys[bp], max_sell, position)
                if qty > 0:
                    orders.append(Order(product, bp, -qty))
                    max_sell -= qty

        # =================================================================
        # PHASE 2 — MAKING (passive, wait for taker bots to hit us)
        #
        # Post limit orders at improved prices to gain queue priority.
        # - Overbid: place our bid 1 tick above the best existing bid
        # - Undercut: place our ask 1 tick below the best existing ask
        # Both must stay on the "correct side" of fair to maintain edge.
        # =================================================================

        # Base prices: just inside the walls
        make_bid = int(bid_wall + 1) if bid_wall is not None else None
        make_ask = int(ask_wall - 1) if ask_wall is not None else None

        # Overbid: find the highest existing bid we can improve on
        if buys and make_bid is not None:
            for bp in sorted(buys.keys(), reverse=True):
                overbid = bp + 1
                if buys[bp] > 1 and overbid < fair:
                    # Overbid by 1 tick — we get priority, still below fair
                    make_bid = max(make_bid, overbid)
                    break
                elif bp < fair:
                    # Volume ≤ 1: match the price instead of overbidding
                    make_bid = max(make_bid, bp)
                    break

        # Undercut: find the lowest existing ask we can improve on
        if sells and make_ask is not None:
            for ap in sorted(sells.keys()):
                undercut = ap - 1
                if sells[ap] > 1 and undercut > fair:
                    make_ask = min(make_ask, undercut)
                    break
                elif ap > fair:
                    make_ask = min(make_ask, ap)
                    break

        # Safety floor/ceiling: guarantee at least 1 full tick of edge
        # This protects against wall_mid estimation error
        if make_bid is not None:
            make_bid = min(make_bid, math.floor(fair) - 1)
        if make_ask is not None:
            make_ask = max(make_ask, math.ceil(fair) + 1)

        # Post passive orders with all remaining capacity
        if make_bid is not None and max_buy > 0:
            orders.append(Order(product, make_bid, max_buy))
        if make_ask is not None and max_sell > 0:
            orders.append(Order(product, make_ask, -max_sell))

        return orders