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
    **{sym: 300 for sym in VOUCHER_STRIKES},
}

# Soft limits (strategy-internal risk throttle)
SOFT_POS_LIMITS: Dict[str, int] = {
    HYDROGEL_SYMBOL: 0,   # intentionally neutral
    UNDERLYING_SYMBOL: 150,
    **{sym: 120 for sym in VOUCHER_STRIKES},
}
SOFT_POS_LIMITS["VEV_5000"] = 150
SOFT_POS_LIMITS["VEV_5100"] = 150
SOFT_POS_LIMITS["VEV_5200"] = 150

# =========================
# Smile coefficients
# sigma(m) = a*m^2 + b*m + c, m = ln(K/S)/sqrt(T)
# =========================
SMILE_A = 0.4887751332
SMILE_B = -0.9894897327
SMILE_C = 0.5143127140

# =========================
# Time (INTENTIONALLY UNCHANGED)
# =========================
DAYS_PER_YEAR = 365.0
INITIAL_TTE_DAYS = 5.0
STEPS_PER_DAY = 10_000
TS_STEP = 100
MIN_TTE_YEARS = 1e-6

# =========================
# IV scalping params
# =========================
THEO_NORM_WINDOW = 20
IV_SCALPING_WINDOW = 100

# Regime gate: trade only in moderate regime
IV_SCALPING_SWITCH_MIN = 0.30
IV_SCALPING_SWITCH_MAX = 2.50

# Spread-aware edge thresholds
OPEN_EDGE_MIN = 0.80
OPEN_EDGE_SPREAD_MULT = 1.00
CLOSE_EDGE_SPREAD_MULT = 0.20

LOW_VEGA_THRESHOLD = 2.0
LOW_VEGA_OPEN_EDGE_ADJ = 0.40

# Trade only stable center strikes
IV_SCALPING_SYMBOLS = [
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
]

# =========================
# Execution / model guards
# =========================
MIN_MODEL_IV = 0.05
MAX_MODEL_IV = 1.50

MAX_OPTION_SPREAD = 3
MAX_UNDERLYING_SPREAD = 5

MAX_OPTION_OPEN_QTY_PER_TICK = 20
MAX_OPTION_CLOSE_QTY_PER_TICK = 40
MAX_UNDERLYING_QTY_PER_TICK = 40
OPTION_MR_MAX_QTY_PER_TICK = 15

# =========================
# Mean reversion params (hybrid)
# =========================
UNDERLYING_MR_WINDOW = 5
UNDERLYING_MR_THR = 20.0

OPTION_MR_WINDOW = 30
OPTION_MR_THR = 2.5
OPTION_MR_SYMBOL: Optional[str] = None  # disabled (was VEV_4000)

USE_FH_UNDERLYING_SIGNAL = True

# =========================
# End of day flatten (INTENTIONALLY UNCHANGED)
# =========================
EOD_FLATTEN_START_STEP = 9800


class BookState:
    def __init__(self, state: TradingState, symbol: str, last_mid: Optional[float] = None):
        self.symbol = symbol

        depth: Optional[OrderDepth] = state.order_depths.get(symbol)
        if depth is None:
            depth = OrderDepth()

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

        # Mid with fallback
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

        # One-sided synthetic fallback (pricing only)
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
        # Kept for template compatibility
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
        # Intentionally unchanged for current 1/10-day backtest setup
        step = self._get_step(timestamp)
        tte_days = INITIAL_TTE_DAYS - (step / STEPS_PER_DAY)
        return max(tte_days / DAYS_PER_YEAR, MIN_TTE_YEARS)

    @staticmethod
    def _smile_iv(S: float, K: float, T: float) -> float:
        if S <= 0 or K <= 0 or T <= 0:
            return SMILE_C
        m = math.log(K / S) / math.sqrt(T)
        iv = SMILE_A * m * m + SMILE_B * m + SMILE_C
        return max(MIN_MODEL_IV, min(MAX_MODEL_IV, iv))

    @staticmethod
    def _bs_call_with_greeks(S: float, K: float, T: float, sigma: float) -> Tuple[float, float, float]:
        sigma = max(sigma, 1e-8)
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

    @staticmethod
    def _open_edge_threshold(spread: float, low_vega_adj: float) -> float:
        return max(OPEN_EDGE_MIN, OPEN_EDGE_SPREAD_MULT * spread) + low_vega_adj

    @staticmethod
    def _close_edge_threshold(spread: float) -> float:
        return max(0.0, CLOSE_EDGE_SPREAD_MULT * spread)

    def _flatten_all(self, contexts: Dict[str, ProductContext]) -> None:
        for ctx in contexts.values():
            b = ctx.book
            if b is None:
                continue
            if ctx.expected_position > 0 and b.has_bid and b.best_bid is not None:
                ctx.sell(b.best_bid, ctx.expected_position)
            elif ctx.expected_position < 0 and b.has_ask and b.best_ask is not None:
                ctx.buy(b.best_ask, -ctx.expected_position)

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

        u_spread = int(u_book.best_ask - u_book.best_bid)
        if u_spread <= 0 or u_spread > MAX_UNDERLYING_SPREAD:
            return

        step = self._get_step(state.timestamp)
        tte = self._tte_years(state.timestamp)

        # Underlying EMAs
        ema_u = self._ema(ema_store, "ema_under_u", u_book.mid, UNDERLYING_MR_WINDOW)
        ema_o = self._ema(ema_store, "ema_under_o", u_book.mid, OPTION_MR_WINDOW)
        ema_u_dev = u_book.mid - ema_u
        ema_o_dev = u_book.mid - ema_o

        under_signal = ema_o_dev if USE_FH_UNDERLYING_SIGNAL else ema_u_dev

        # Underlying midpoint used for option theo valuation
        S_theo = 0.5 * (u_book.best_bid + u_book.best_ask)

        indicators: Dict[str, Dict[str, float]] = {}
        for sym, K in VOUCHER_STRIKES.items():
            o_ctx = contexts.get(sym)
            if o_ctx is None or o_ctx.book is None:
                continue

            b = o_ctx.book
            if b.mid <= 0 or not (b.has_bid and b.has_ask):
                continue

            spread = float(b.best_ask - b.best_bid)
            if spread <= 0 or spread > MAX_OPTION_SPREAD:
                continue

            iv = self._smile_iv(S_theo, float(K), tte)
            theo, delta, vega = self._bs_call_with_greeks(S_theo, float(K), tte, iv)

            theo_diff = b.mid - theo
            mean_diff = self._ema(ema_store, f"{sym}_mean_diff", theo_diff, THEO_NORM_WINDOW)
            switch = self._ema(
                ema_store,
                f"{sym}_switch",
                abs(theo_diff - mean_diff),
                IV_SCALPING_WINDOW,
            )

            fair = theo + mean_diff
            sell_edge = float(b.best_bid) - fair   # favorable short when positive/large
            buy_edge = fair - float(b.best_ask)    # favorable long when positive/large

            indicators[sym] = {
                "theo": theo,
                "delta": delta,
                "vega": vega,
                "theo_diff": theo_diff,
                "mean_diff": mean_diff,
                "switch": switch,
                "spread": spread,
                "sell_edge": sell_edge,
                "buy_edge": buy_edge,
            }

        # 1) IV scalping on selected strikes
        if step >= max(THEO_NORM_WINDOW, IV_SCALPING_WINDOW):
            for sym in IV_SCALPING_SYMBOLS:
                ctx = contexts.get(sym)
                if ctx is None or ctx.book is None or sym not in indicators:
                    continue

                b = ctx.book
                ind = indicators[sym]

                regime_active = (
                    ind["switch"] >= IV_SCALPING_SWITCH_MIN
                    and ind["switch"] <= IV_SCALPING_SWITCH_MAX
                )

                # Regime inactive -> reduce risk gradually
                if not regime_active:
                    pos = ctx.expected_position
                    if pos > 0:
                        ctx.sell(b.best_bid, min(MAX_OPTION_CLOSE_QTY_PER_TICK, pos))
                    elif pos < 0:
                        ctx.buy(b.best_ask, min(MAX_OPTION_CLOSE_QTY_PER_TICK, -pos))
                    continue

                low_vega_adj = LOW_VEGA_OPEN_EDGE_ADJ if ind["vega"] <= LOW_VEGA_THRESHOLD else 0.0
                open_thr = self._open_edge_threshold(ind["spread"], low_vega_adj)
                close_thr = self._close_edge_threshold(ind["spread"])

                sell_edge = ind["sell_edge"]
                buy_edge = ind["buy_edge"]

                # Close first; avoid same-tick hard flips
                closed_this_tick = False
                pos = ctx.expected_position
                if pos > 0 and sell_edge >= close_thr:
                    ctx.sell(b.best_bid, min(MAX_OPTION_CLOSE_QTY_PER_TICK, pos))
                    closed_this_tick = True
                elif pos < 0 and buy_edge >= close_thr:
                    ctx.buy(b.best_ask, min(MAX_OPTION_CLOSE_QTY_PER_TICK, -pos))
                    closed_this_tick = True

                if closed_this_tick:
                    continue

                # Open/add in direction of edge
                pos = ctx.expected_position
                if sell_edge >= open_thr and pos <= 0 and ctx.remaining_sell > 0:
                    ctx.sell(b.best_bid, min(MAX_OPTION_OPEN_QTY_PER_TICK, ctx.remaining_sell))
                elif buy_edge >= open_thr and pos >= 0 and ctx.remaining_buy > 0:
                    ctx.buy(b.best_ask, min(MAX_OPTION_OPEN_QTY_PER_TICK, ctx.remaining_buy))

        # 2) Optional option MR overlay (disabled by default)
        if (
            OPTION_MR_SYMBOL is not None
            and step >= OPTION_MR_WINDOW
            and OPTION_MR_SYMBOL in indicators
            and OPTION_MR_SYMBOL in contexts
        ):
            ctx = contexts[OPTION_MR_SYMBOL]
            b = ctx.book
            ind = indicators[OPTION_MR_SYMBOL]

            if b is not None and b.has_bid and b.has_ask:
                mr_signal = ema_o_dev + (ind["theo_diff"] - ind["mean_diff"])
                mr_thr = max(OPTION_MR_THR, 0.5 * ind["spread"])

                pos = ctx.expected_position
                if mr_signal > mr_thr:
                    if pos > 0:
                        ctx.sell(b.best_bid, min(OPTION_MR_MAX_QTY_PER_TICK, pos))
                    else:
                        ctx.sell(b.best_bid, min(OPTION_MR_MAX_QTY_PER_TICK, ctx.remaining_sell))
                elif mr_signal < -mr_thr:
                    if pos < 0:
                        ctx.buy(b.best_ask, min(OPTION_MR_MAX_QTY_PER_TICK, -pos))
                    else:
                        ctx.buy(b.best_ask, min(OPTION_MR_MAX_QTY_PER_TICK, ctx.remaining_buy))

        # 3) Underlying mean reversion (capped size, no hard flips)
        if step >= UNDERLYING_MR_WINDOW:
            pos = u_ctx.expected_position

            if under_signal > UNDERLYING_MR_THR:
                if pos > 0:
                    u_ctx.sell(u_book.best_bid, min(MAX_UNDERLYING_QTY_PER_TICK, pos))
                else:
                    u_ctx.sell(u_book.best_bid, min(MAX_UNDERLYING_QTY_PER_TICK, u_ctx.remaining_sell))

            elif under_signal < -UNDERLYING_MR_THR:
                if pos < 0:
                    u_ctx.buy(u_book.best_ask, min(MAX_UNDERLYING_QTY_PER_TICK, -pos))
                else:
                    u_ctx.buy(u_book.best_ask, min(MAX_UNDERLYING_QTY_PER_TICK, u_ctx.remaining_buy))

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
            b = BookState(state, sym, last_mid.get(sym))
            books[sym] = b

            pos = int(state.position.get(sym, 0))
            hard_limit = int(POS_LIMITS.get(sym, 0))
            soft_limit = int(SOFT_POS_LIMITS.get(sym, hard_limit))
            limit = min(hard_limit, max(0, soft_limit))

            contexts[sym] = ProductContext(sym, pos, limit, b)

            if b.mid > 0:
                last_mid[sym] = b.mid

        step = self._get_step(state.timestamp)

        # EOD flatten threshold intentionally unchanged
        if step >= EOD_FLATTEN_START_STEP:
            self._flatten_all(contexts)
        else:
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