from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json


class Trader:
    """
    Strategy logic (derived from your analytics):
    - Improve by 1 tick only where maker_improve1 stays robustly positive.
    - Touch-only where improve1 is negative.
    - VELVETFRUIT_EXTRACT: side-specific blackout after Mark67 events.
    - Inventory-aware size skew + hard caps.
    - End-of-day flattening.
    """

    # IMPORTANT: replace pos_limit with official limits from your competition spec
    CONFIG = {
        "HYDROGEL_PACK": {"pos_limit": 140, "size": 20, "improve": 1},
        "VEV_4000": {"pos_limit": 110, "size": 16, "improve": 1},
        "VELVETFRUIT_EXTRACT": {"pos_limit": 160, "size": 18, "improve": 1},

        # Light/touch-only legs
        "VEV_5200": {"pos_limit": 70, "size": 6, "improve": 0},
        "VEV_5300": {"pos_limit": 70, "size": 6, "improve": 0},
        "VEV_5400": {"pos_limit": 70, "size": 6, "improve": 0},
        "VEV_5500": {"pos_limit": 70, "size": 5, "improve": 0},
        "VEV_6000": {"pos_limit": 40, "size": 3, "improve": 0},
        "VEV_6500": {"pos_limit": 40, "size": 3, "improve": 0},
    }

    # Mark67 blackout on VFE:
    # W=3 chosen as best practical tradeoff in your sweep.
    BLACKOUT_STEPS = 3
    STEP_TS = 100
    BLACKOUT_TS = BLACKOUT_STEPS * STEP_TS

    # Inventory controls
    INV_SOFT_FRAC = 0.60
    INV_HARD_FRAC = 0.90

    # End-of-day risk controls
    FLATTEN_START = 980000
    FLATTEN_CLIP = 25

    def run(self, state: TradingState):
        mem = self._load_memory(state.traderData)

        # New-day detection (timestamp reset)
        if mem["last_ts"] > state.timestamp:
            mem["m67_buy_ts"] = -10**12
            mem["m67_sell_ts"] = -10**12

        self._update_mark67_memory(state, mem)

        result: Dict[str, List[Order]] = {}
        flatten_mode = state.timestamp >= self.FLATTEN_START

        for symbol, cfg in self.CONFIG.items():
            depth = state.order_depths.get(symbol)
            if depth is None:
                continue

            pos = state.position.get(symbol, 0)

            if flatten_mode:
                result[symbol] = self._flatten_symbol(symbol, depth, pos)
            else:
                result[symbol] = self._quote_symbol(
                    symbol=symbol,
                    depth=depth,
                    pos=pos,
                    ts=state.timestamp,
                    cfg=cfg,
                    mem=mem,
                )

        mem["last_ts"] = state.timestamp
        trader_data_out = json.dumps(mem)
        conversions = 0

        # If your environment expects only "result", change this return accordingly.
        return result, conversions, trader_data_out

    # -------------------- helpers --------------------

    def _load_memory(self, trader_data: str) -> Dict[str, int]:
        mem = {"last_ts": -1, "m67_buy_ts": -10**12, "m67_sell_ts": -10**12}
        if trader_data:
            try:
                raw = json.loads(trader_data)
                for k in mem:
                    if k in raw:
                        mem[k] = int(raw[k])
            except Exception:
                pass
        return mem

    def _update_mark67_memory(self, state: TradingState, mem: Dict[str, int]) -> None:
        trades = state.market_trades.get("VELVETFRUIT_EXTRACT", [])
        for tr in trades:
            tts = getattr(tr, "timestamp", state.timestamp)
            buyer = getattr(tr, "buyer", "")
            seller = getattr(tr, "seller", "")

            if buyer == "Mark 67":
                mem["m67_buy_ts"] = max(mem["m67_buy_ts"], int(tts))
            if seller == "Mark 67":
                mem["m67_sell_ts"] = max(mem["m67_sell_ts"], int(tts))

    def _best_bid_ask(self, depth: OrderDepth):
        best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return best_bid, best_ask

    def _quote_symbol(
        self,
        symbol: str,
        depth: OrderDepth,
        pos: int,
        ts: int,
        cfg: Dict[str, int],
        mem: Dict[str, int],
    ) -> List[Order]:
        best_bid, best_ask = self._best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return []

        spread = best_ask - best_bid
        if spread <= 0:
            return []

        limit = cfg["pos_limit"]
        base_size = cfg["size"]
        improve = cfg["improve"]

        # Base quote prices
        bid_px = best_bid
        ask_px = best_ask

        # Improve only where research supports it
        if improve > 0:
            # Improve both sides only if quote won't lock/cross itself
            if spread >= 2 * improve + 1:
                bid_px = best_bid + improve
                ask_px = best_ask - improve
            # Otherwise improve one side that helps inventory rebalance
            elif spread >= improve + 1:
                if pos < 0:
                    bid_px = best_bid + improve
                elif pos > 0:
                    ask_px = best_ask - improve

        # Capacity wrt limits
        buy_cap = max(0, limit - pos)
        sell_cap = max(0, limit + pos)

        bid_qty = min(base_size, buy_cap)
        ask_qty = min(base_size, sell_cap)

        # Inventory-aware skew
        soft = int(self.INV_SOFT_FRAC * limit)
        hard = int(self.INV_HARD_FRAC * limit)

        if pos > soft:
            # long: suppress bids, encourage asks
            bid_qty = min(bid_qty, max(1, base_size // 4))
            ask_qty = min(sell_cap, base_size + base_size // 2)
        elif pos < -soft:
            # short: suppress asks, encourage bids
            ask_qty = min(ask_qty, max(1, base_size // 4))
            bid_qty = min(buy_cap, base_size + base_size // 2)

        if pos >= hard:
            bid_qty = 0
        if pos <= -hard:
            ask_qty = 0

        # VFE toxicity filter (side-specific)
        if symbol == "VELVETFRUIT_EXTRACT":
            # Mark67 buy aggression -> avoid selling into likely continuation
            if ts - mem["m67_buy_ts"] <= self.BLACKOUT_TS:
                ask_qty = 0
            # Mark67 sell aggression -> avoid buying into likely continuation
            if ts - mem["m67_sell_ts"] <= self.BLACKOUT_TS:
                bid_qty = 0

        # Avoid crossed/locked own quotes
        if bid_qty > 0 and ask_qty > 0 and bid_px >= ask_px:
            if pos >= 0:
                bid_qty = 0
            else:
                ask_qty = 0

        # Clamp to passive levels
        if bid_qty > 0:
            bid_px = min(bid_px, best_ask - 1)
        if ask_qty > 0:
            ask_px = max(ask_px, best_bid + 1)

        orders: List[Order] = []
        if bid_qty > 0:
            orders.append(Order(symbol, int(bid_px), int(bid_qty)))
        if ask_qty > 0:
            orders.append(Order(symbol, int(ask_px), -int(ask_qty)))
        return orders

    def _flatten_symbol(self, symbol: str, depth: OrderDepth, pos: int) -> List[Order]:
        best_bid, best_ask = self._best_bid_ask(depth)
        orders: List[Order] = []

        if pos > 0 and best_bid is not None:
            qty = min(pos, self.FLATTEN_CLIP)
            if qty > 0:
                orders.append(Order(symbol, best_bid, -qty))
        elif pos < 0 and best_ask is not None:
            qty = min(-pos, self.FLATTEN_CLIP)
            if qty > 0:
                orders.append(Order(symbol, best_ask, qty))

        return orders