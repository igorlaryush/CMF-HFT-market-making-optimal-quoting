from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict


@dataclass
class Order:  # Our own placed order
    place_ts: float  # ts when we place the order
    exchange_ts: float  # ts when exchange(simulator) get the order
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class CancelOrder:
    exchange_ts: float
    id_to_delete: int


@dataclass
class AnonTrade:  # Market trade
    exchange_ts: float
    receive_ts: float
    side: str
    size: float
    price: float


@dataclass
class OwnTrade:  # Execution of own placed order
    place_ts: float  # ts when we call place_order method, for debugging
    exchange_ts: float
    receive_ts: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float
    execute: str  # BOOK or TRADE

    def __post_init__(self):
        assert isinstance(self.side, str)


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    exchange_ts: float
    receive_ts: float
    asks: List[Tuple[float, float]]  # tuple[price, size]
    bids: List[Tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    exchange_ts: float
    receive_ts: float
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trade: Optional[AnonTrade] = None


def get_pnl(updates_list: List[Union[MdUpdate, OwnTrade]], cost=-0.00001) -> pd.DataFrame:
    '''
        This function calculates PnL from list of updates
    '''

    # current position in btc and usd
    btc_pos, usd_pos = 0.0, 0.0

    N = len(updates_list)
    btc_pos_arr = np.zeros((N, ))
    usd_pos_arr = np.zeros((N, ))
    mid_price_arr = np.zeros((N, ))
    # current best_bid and best_ask
    best_bid: float = -np.inf
    best_ask: float = np.inf

    for i, update in enumerate(updates_list):

        if isinstance(update, MdUpdate):
            best_bid, best_ask = update_best_positions(
                best_bid, best_ask, update)
        # mid price
        # i use it to calculate current portfolio value
        mid_price = 0.5 * (best_ask + best_bid)

        if isinstance(update, OwnTrade):
            trade = update
            # update positions
            if trade.side == 'BID':
                btc_pos += trade.size
                usd_pos -= trade.price * trade.size
            elif trade.side == 'ASK':
                btc_pos -= trade.size
                usd_pos += trade.price * trade.size
            usd_pos -= cost * trade.price * trade.size
        # current portfolio value

        btc_pos_arr[i] = btc_pos
        usd_pos_arr[i] = usd_pos
        mid_price_arr[i] = mid_price

    worth_arr = btc_pos_arr * mid_price_arr + usd_pos_arr
    receive_ts = [update.receive_ts for update in updates_list]
    exchange_ts = [update.exchange_ts for update in updates_list]

    df = pd.DataFrame({"exchange_ts": exchange_ts, "receive_ts": receive_ts, "total": worth_arr, "BTC": btc_pos_arr,
                       "USD": usd_pos_arr, "mid_price": mid_price_arr})
    return df


def trade_to_dataframe(trades_list: List[OwnTrade]) -> pd.DataFrame:
    exchange_ts = [trade.exchange_ts for trade in trades_list]
    receive_ts = [trade.receive_ts for trade in trades_list]

    size = [trade.size for trade in trades_list]
    price = [trade.price for trade in trades_list]
    side = [trade.side for trade in trades_list]

    dct = {
        "exchange_ts": exchange_ts,
        "receive_ts": receive_ts,
        "size": size,
        "price": price,
        "side": side
    }

    df = pd.DataFrame(dct).groupby('receive_ts').agg(
        lambda x: x.iloc[-1]).reset_index()
    return df


def md_to_dataframe(md_list: List[MdUpdate]) -> pd.DataFrame:

    best_bid = -np.inf
    best_ask = np.inf
    best_bids = []
    best_asks = []
    for md in md_list:
        best_bid, best_ask = update_best_positions(best_bid, best_ask, md)

        best_bids.append(best_bid)
        best_asks.append(best_ask)

    exchange_ts = [md.exchange_ts for md in md_list]
    receive_ts = [md.receive_ts for md in md_list]
    dct = {
        "exchange_ts": exchange_ts,
        "receive_ts": receive_ts,
        "bid_price": best_bids,
        "ask_price": best_asks
    }

    df = pd.DataFrame(dct).groupby('receive_ts').agg(
        lambda x: x.iloc[-1]).reset_index()
    return df


def update_best_positions(best_bid: float, best_ask: float, md: MdUpdate) -> Tuple[float, float]:
    if not md.orderbook is None:
        best_bid = md.orderbook.bids[0][0]
        best_ask = md.orderbook.asks[0][0]
    elif not md.trade is None:
        if md.trade.side == 'BID':
            best_ask = max(md.trade.price, best_ask)
        elif md.trade.side == 'ASK':
            best_bid = min(best_bid, md.trade.price)
        else:
            assert False, "WRONG TRADE SIDE"
    assert best_ask > best_bid, "wrong best positions"
    return best_bid, best_ask


def get_mid_price(mid_price: float, md: MdUpdate):
    book = md.orderbook
    if book is None:
        return mid_price

    price = 0.0
    pos = 0.0

    for i in range(len(book.asks)):
        price += book.asks[i][0] * book.asks[i][1]
        pos += book.asks[i][1]
        price += book.bids[i][0] * book.bids[i][1]
        pos += book.bids[i][1]
    price /= pos
    return price


class PriorQueue:
    def __init__(self, default_key=np.inf, default_val=None):
        self._queue = SortedDict()
        self._min_key = np.inf

    def push(self, key, val):
        if key not in self._queue:
            self._queue[key] = []
        self._queue[key].append(val)
        self._min_key = min(self._min_key, key)

    def pop(self):
        if len(self._queue) == 0:
            return np.inf, None
        res = self._queue.popitem(0)
        self._min_key = np.inf
        if len(self._queue):
            self._min_key = self._queue.peekitem(0)[0]
        return res

    def min_key(self):
        return self._min_key


class PriorHeap():
    def __init__(self, *args, **kwargs):
        self._map = {}
        self._heap = SortedDict()

    def __str__(self):
        return "map: " + str(self._map) + "\nheap: " + str(self._heap)

    def push(self, order_id: int, order_price: float):
        # add order to map
        self._map[order_id] = order_price

        if order_price not in self._heap.keys():
            self._heap[order_price] = set()
        self._heap[order_price].add(order_id)

    def erase(self, order_id):
        if not order_id in self._map:
            return False
        order_price = self._map.pop(order_id)
        self._heap[order_price].remove(order_id)
        if len(self._heap[order_price]) == 0:
            self._heap.pop(order_price)
        return True

    def greater_or_eq(self, price: float) -> Set[int]:
        ind = self._heap.bisect_left(price)
        res = []
        for _price in self._heap.keys()[ind:]:
            res += self._heap[_price]
        return res

    def less_or_eq(self, price: float) -> Set[int]:
        ind = self._heap.bisect_right(price)
        res = []
        for _price in self._heap.keys()[:ind]:
            res += self._heap[_price]
        return res
