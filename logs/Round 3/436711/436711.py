from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math
from statistics import NormalDist

_N = NormalDist()

# =========================
# Symbols
# =========================
HYDROGEL_SYMBOL = "HYDROGEL_PACK"
UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"

VOUCHER_STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

ALL_SYMBOLS = [HYDROGEL_SYMBOL, UNDERLYING_SYMBOL] + list(VOUCHER_STRIKES.keys())

# =========================
# Position limits
# =========================
POS_LIMITS: Dict[str, int] = {
    HYDROGEL_SYMBOL: 200,
    UNDERLYING_SYMBOL: 200,
    **{sym: 300 for sym in VOUCHER_STRIKES}
}

# =========================
# Smile coefficients
# sigma(m) = a*m^2 + b*m + c, m = ln(K/S)/sqrt(T)
# =========================
SMILE_A = 0.4887751332
SMILE_B = -0.9894897327
SMILE_C = 0.5143127140

# =========================
# Time
# =========================
DAYS_PER_YEAR = 365.0
INITIAL_TTE_DAYS = 5.0        # round 3 start
STEPS_PER_DAY = 10_000        # timestamp//100 runs 0..9999
TS_STEP = 100
MIN_TTE_YEARS = 1e-6

# =========================
# IV scalping params (FH-style)
# =========================
THEO_NORM_WINDOW = 20
IV_SCALPING_WINDOW = 100
IV_SCALPING_SWITCH_MIN = 0.7

THR_OPEN = 0.5
THR_CLOSE = 0.0
LOW_VEGA_THRESHOLD = 1.0
LOW_VEGA_THR_ADJ = 0.5

# Trade only subset for IV scalping
IV_SCALPING_SYMBOLS = [
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
]

# =========================
# Mean reversion params (FH-style hybrid)
# =========================
UNDERLYING_MR_WINDOW = 10
UNDERLYING_MR_THR = 15.0

OPTION_MR_WINDOW = 30
OPTION_MR_THR = 5.0
OPTION_MR_SYMBOL = "VEV_4000"   # deep ITM proxy

# If True, replicate FH behavior where underlying signal uses "option-window EMA deviation"
USE_FH_UNDERLYING_SIGNAL = True

# =========================
# End of day flatten
# =========================
EOD_FLATTEN_START_STEP = 9800  # last 200 steps


class BookState:
    def __init__(self, state: TradingState, symbol: str, last_mid: Optional[float] = None):
        self.symbol = symbol

        depth: Optional[OrderDepth] = state.order_depths.get(symbol)
        if depth is None:
            depth = OrderDepth()

        # Normalize depth to positive volumes
        self.buy_levels: Dict[int, int] = {}
        for p, v in depth.buy_orders.items():
            if v == 0:
                continue
            self.buy_levels[int(p)] = int(abs(v))

        self.sell_levels: Dict[int, int] = {}
        for p, v in depth.sell_orders.items():
            if v == 0:
                continue
            self.sell_levels[int(p)] = int(abs(v))

        self.has_bid = len(self.buy_levels) > 0
        self.has_ask = len(self.sell_levels) > 0

        self.best_bid: Optional[int] = max(self.buy_levels.keys()) if self.has_bid else None
        self.best_ask: Optional[int] = min(self.sell_levels.keys()) if self.has_ask else None

        self.bid_wall: Optional[int] = min(self.buy_levels.keys()) if self.has_bid else None
        self.ask_wall: Optional[int] = max(self.sell_levels.keys()) if self.has_ask else None

        # Mid logic with fallback (handles no-book => mid=0 situation)
        if self.best_bid is not None and self.best_ask is not None:
            self.mid = 0.5 * (self.best_bid + self.best_ask)
        elif self.best_bid is not None:
            self.mid = self.best_bid + 0.5
        elif self.best_ask is not None:
            self.mid = self.best_ask - 0.5
        elif last_mid is not None and last_mid > 0:
            self.mid = float(last_mid)
        else:
            self.mid = 0.0

        # Synthetic one-sided fallback for pricing only
        if self.best_bid is None and self.best_ask is not None:
            self.best_bid = self.best_ask - 1
        if self.best_ask is None and self.best_bid is not None:
            self.best_ask = self.best_bid + 1


class ProductContext:
    def __init__(self, symbol: str, position: int, limit: int, book: Optional[BookState]):
        self.symbol = symbol
        self.position = int(position)
        self.expected_position = int(position)
        self.limit = int(limit)
        self.book = book
        self.orders: List[Order] = []

        self.remaining_buy = self.limit - self.position
        self.remaining_sell = self.limit + self.position

    def buy(self, price: Optional[int], qty: int) -> None:
        if price is None:
            return
        q = min(max(int(qty), 0), self.remaining_buy)
        if q <= 0:
            return
        self.orders.append(Order(self.symbol, int(price), int(q)))
        self.remaining_buy -= q
        self.expected_position += q

    def sell(self, price: Optional[int], qty: int) -> None:
        if price is None:
            return
        q = min(max(int(qty), 0), self.remaining_sell)
        if q <= 0:
            return
        self.orders.append(Order(self.symbol, int(price), -int(q)))
        self.remaining_sell -= q
        self.expected_position -= q


class Trader:
    def bid(self):
        # Ignored for round 3, kept for compatibility with template
        return 15

    @staticmethod
    def _safe_load_trader_data(raw: str) -> Dict:
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    @staticmethod
    def _ema(store: Dict[str, float], key: str, value: float, window: int) -> float:
        if window <= 1:
            store[key] = value
            return value
        alpha = 2.0 / (window + 1.0)
        prev = store.get(key, value)
        new_val = alpha * value + (1.0 - alpha) * prev
        store[key] = new_val
        return new_val

    @staticmethod
    def _get_step(timestamp: int) -> int:
        return int(timestamp // TS_STEP)

    def _tte_years(self, timestamp: int) -> float:
        # FH-style continuous intra-day decay
        step = self._get_step(timestamp)
        tte_days = INITIAL_TTE_DAYS - (step / STEPS_PER_DAY)
        return max(tte_days / DAYS_PER_YEAR, MIN_TTE_YEARS)

    @staticmethod
    def _smile_iv(S: float, K: float, T: float) -> float:
        m = math.log(K / S) / math.sqrt(T)
        return SMILE_A * m * m + SMILE_B * m + SMILE_C

    @staticmethod
    def _bs_call_with_greeks(S: float, K: float, T: float, sigma: float) -> Tuple[float, float, float]:
        sigma = max(sigma, 1e-8)  # numerical guard
        if S <= 0 or K <= 0 or T <= 0:
            intrinsic = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
            return intrinsic, delta, 0.0

        sqrtT = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT

        call = S * _N.cdf(d1) - K * _N.cdf(d2)
        delta = _N.cdf(d1)
        vega = S * _N.pdf(d1) * sqrtT
        return call, delta, vega

    def _flatten_all(self, contexts: Dict[str, ProductContext]) -> None:
        for ctx in contexts.values():
            b = ctx.book
            if b is None:
                continue
            if ctx.position > 0 and b.has_bid and b.best_bid is not None:
                ctx.sell(b.best_bid, ctx.position)
            elif ctx.position < 0 and b.has_ask and b.best_ask is not None:
                ctx.buy(b.best_ask, -ctx.position)

    def _trade_hybrid_vev(
        self,
        state: TradingState,
        contexts: Dict[str, ProductContext],
        ema_store: Dict[str, float],
    ) -> None:
        u_ctx = contexts.get(UNDERLYING_SYMBOL)
        if u_ctx is None or u_ctx.book is None:
            return

        u_book = u_ctx.book
        if not (u_book.has_bid and u_book.has_ask) or u_book.mid <= 0:
            return

        step = self._get_step(state.timestamp)
        tte = self._tte_years(state.timestamp)

        # Underlying EMAs
        ema_u = self._ema(ema_store, "ema_under_u", u_book.mid, UNDERLYING_MR_WINDOW)
        ema_o = self._ema(ema_store, "ema_under_o", u_book.mid, OPTION_MR_WINDOW)
        ema_u_dev = u_book.mid - ema_u
        ema_o_dev = u_book.mid - ema_o

        under_signal = ema_o_dev if USE_FH_UNDERLYING_SIGNAL else ema_u_dev

        # Underlying mid used in option theo valuation
        S_theo = 0.5 * (u_book.best_bid + u_book.best_ask)

        indicators: Dict[str, Dict[str, float]] = {}
        for sym, K in VOUCHER_STRIKES.items():
            o_ctx = contexts.get(sym)
            if o_ctx is None or o_ctx.book is None:
                continue
            b = o_ctx.book
            if b.mid <= 0 or not (b.has_bid and b.has_ask):
                continue

            iv = self._smile_iv(S_theo, float(K), tte)
            theo, delta, vega = self._bs_call_with_greeks(S_theo, float(K), tte, iv)

            theo_diff = b.mid - theo
            mean_diff = self._ema(ema_store, f"{sym}_mean_diff", theo_diff, THEO_NORM_WINDOW)
            switch = self._ema(
                ema_store,
                f"{sym}_switch",
                abs(theo_diff - mean_diff),
                IV_SCALPING_WINDOW
            )

            indicators[sym] = {
                "theo": theo,
                "delta": delta,
                "vega": vega,
                "theo_diff": theo_diff,
                "mean_diff": mean_diff,
                "switch": switch,
            }

        # 1) IV Scalping on subset
        if step >= max(THEO_NORM_WINDOW, IV_SCALPING_WINDOW):
            for sym in IV_SCALPING_SYMBOLS:
                if sym not in indicators or sym not in contexts:
                    continue

                ctx = contexts[sym]
                b = ctx.book
                ind = indicators[sym]

                if b is None or not (b.has_bid and b.has_ask):
                    continue

                if ind["switch"] >= IV_SCALPING_SWITCH_MIN:
                    low_vega_adj = LOW_VEGA_THR_ADJ if ind["vega"] <= LOW_VEGA_THRESHOLD else 0.0

                    # FH-style signal structure
                    sell_signal = (b.best_bid - ind["theo"]) - ind["mean_diff"]
                    buy_signal = (b.best_ask - ind["theo"]) - ind["mean_diff"]

                    # Open
                    if sell_signal >= (THR_OPEN + low_vega_adj) and ctx.remaining_sell > 0:
                        ctx.sell(b.best_bid, ctx.remaining_sell)
                    elif buy_signal <= -(THR_OPEN + low_vega_adj) and ctx.remaining_buy > 0:
                        ctx.buy(b.best_ask, ctx.remaining_buy)

                    # Close
                    if ctx.position > 0 and sell_signal >= THR_CLOSE:
                        ctx.sell(b.best_bid, ctx.position)
                    elif ctx.position < 0 and buy_signal <= -THR_CLOSE:
                        ctx.buy(b.best_ask, -ctx.position)
                else:
                    # Regime inactive -> flatten
                    if ctx.position > 0:
                        ctx.sell(b.best_bid, ctx.position)
                    elif ctx.position < 0:
                        ctx.buy(b.best_ask, -ctx.position)

        # 2) Option mean reversion on deep ITM
        if step >= OPTION_MR_WINDOW and OPTION_MR_SYMBOL in indicators and OPTION_MR_SYMBOL in contexts:
            ctx = contexts[OPTION_MR_SYMBOL]
            b = ctx.book
            ind = indicators[OPTION_MR_SYMBOL]

            if b is not None and b.has_bid and b.has_ask:
                mr_signal = ema_o_dev + (ind["theo_diff"] - ind["mean_diff"])

                if mr_signal > OPTION_MR_THR and ctx.remaining_sell > 0:
                    ctx.sell(b.best_bid, ctx.remaining_sell)
                elif mr_signal < -OPTION_MR_THR and ctx.remaining_buy > 0:
                    ctx.buy(b.best_ask, ctx.remaining_buy)

        # 3) Underlying mean reversion
        if step >= UNDERLYING_MR_WINDOW:
            if under_signal > UNDERLYING_MR_THR and u_ctx.remaining_sell > 0:
                u_ctx.sell(u_book.best_bid, u_ctx.remaining_sell)
            elif under_signal < -UNDERLYING_MR_THR and u_ctx.remaining_buy > 0:
                u_ctx.buy(u_book.best_ask, u_ctx.remaining_buy)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        td = self._safe_load_trader_data(state.traderData)
        ema_store = td.get("ema", {})
        if not isinstance(ema_store, dict):
            ema_store = {}

        last_mid = td.get("last_mid", {})
        if not isinstance(last_mid, dict):
            last_mid = {}

        books: Dict[str, BookState] = {}
        contexts: Dict[str, ProductContext] = {}

        for sym in ALL_SYMBOLS:
            if sym not in state.order_depths:
                continue
            b = BookState(state, sym, last_mid.get(sym))
            books[sym] = b
            pos = state.position.get(sym, 0)
            limit = POS_LIMITS.get(sym, 0)
            contexts[sym] = ProductContext(sym, pos, limit, b)

            if b.mid > 0:
                last_mid[sym] = b.mid

        step = self._get_step(state.timestamp)

        # EOD risk-off
        if step >= EOD_FLATTEN_START_STEP:
            self._flatten_all(contexts)
        else:
            # HYDROGEL intentionally neutral for now
            self._trade_hybrid_vev(state, contexts, ema_store)

        for sym, ctx in contexts.items():
            if ctx.orders:
                result[sym] = ctx.orders

        out_td = {"ema": ema_store, "last_mid": last_mid}
        try:
            trader_data = json.dumps(out_td, separators=(",", ":"))
        except Exception:
            trader_data = ""

        return result, conversions, trader_data