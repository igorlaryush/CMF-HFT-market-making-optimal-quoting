from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simulator.simulator import (MdUpdate, Order, OwnTrade, Sim,
                                 update_best_positions)


class LimitMarketStrategy:
    """
        This strategy places limit or market orders every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    """

    def __init__(
            self,
            line_coefficients: Tuple[float, float],
            parabola_coefficients: Tuple[float, float, float],
            trade_size: Optional[float] = 0.001,
            price_tick: Optional[float] = 0.1,
            delay: Optional[int] = 1e8,
            hold_time: Optional[int] = 1e10
    ) -> None:
        """
            Args:
                line_coefficients: line coefficients [k, b] y = kx + b
                parabola_coefficients: parabola coefficients [a, b, c] y = ax^2 + bx + c
                trade_size: volume of each trade
                price_tick: a value by which we increase a bid (reduce an ask) limit order
                delay: delay between orders in nanoseconds
                hold_time: holding time in nanoseconds
        """

        self.trade_size = trade_size
        self.delay = delay
        if hold_time is None:
            hold_time = min(delay * 5, pd.Timedelta(10, 's').delta)
        self.hold_time = hold_time

        # market data list
        self.md_list = []
        # executed trades list
        self.trades_list = []
        # all updates list
        self.updates_list = []

        self.current_time = None
        self.coin_position = 0
        self.prev_midprice = None
        self.current_midprice = None
        self.current_spread = None
        self.price_tick = price_tick

        self.line_k, self.line_b = line_coefficients
        self.parabola_a, self.parabola_b, self.parabola_c = parabola_coefficients

        self.actions_history = []

    def get_normalized_data(self) -> Tuple[float, float]:
        # implement normalization
        return self.coin_position, self.current_spread

    def run(self, sim: Sim) -> \
            Tuple[List[OwnTrade], List[MdUpdate], List[Union[OwnTrade, MdUpdate]], List[Order]]:
        """
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates
                received by strategy(market data and information about executed trades)
                actions_history: list of tuples(time, coin_pos, spread, action)
        """

        # current best positions
        best_bid = -np.inf
        best_ask = np.inf

        # last order timestamp
        prev_time = -np.inf
        # orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        while True:
            # get update from simulator
            self.current_time, updates = sim.tick()
            if updates is None:
                break
            # save updates
            self.updates_list += updates
            for update in updates:
                # update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(
                        best_bid, best_ask, update)
                    self.md_list.append(update)
                    self.current_spread = best_ask - best_bid

                elif isinstance(update, OwnTrade):
                    self.trades_list.append(update)
                    # delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)

                    if update.side == 'BID':
                        self.coin_position += update.size
                    else:
                        self.coin_position -= update.size
                else:
                    assert False, 'invalid type of update!'

            if self.current_time - prev_time >= self.delay:
                # place order
                inventory, spread = self.get_normalized_data()

                if (self.parabola_a * inventory ** 2 + self.parabola_b * inventory + self.parabola_c) > spread:
                    bid_market_order = sim.place_order(
                        self.current_time, self.trade_size, 'BID', best_ask)
                    ongoing_orders[bid_market_order.order_id] = bid_market_order
                    action = 'market buy'
                elif (self.parabola_a * inventory ** 2 + self.parabola_b * (-inventory) + self.parabola_c) > spread:
                    ask_market_order = sim.place_order(
                        self.current_time, self.trade_size, 'ASK', best_bid)
                    ongoing_orders[ask_market_order.order_id] = ask_market_order
                    action = 'market sell'
                else:
                    above_line1 = (self.line_k * inventory +
                                   self.line_b) < spread
                    above_line2 = (self.line_k * (-inventory) +
                                   self.line_b) < spread

                    bid_price = best_bid + self.price_tick * above_line1
                    ask_price = best_ask - self.price_tick * above_line2

                    bid_limit_order = sim.place_order(
                        self.current_time, self.trade_size, 'BID', bid_price)
                    ask_limit_order = sim.place_order(
                        self.current_time, self.trade_size, 'ASK', ask_price)
                    ongoing_orders[bid_limit_order.order_id] = bid_limit_order
                    ongoing_orders[ask_limit_order.order_id] = ask_limit_order
                    action = 'limit order'

                prev_time = self.current_time
                self.actions_history.append((self.current_time, self.coin_position,
                                             self.current_spread, action))

            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < self.current_time - self.hold_time:
                    sim.cancel_order(self.current_time, ID)
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)

        return self.trades_list, self.md_list, self.updates_list, self.actions_history
