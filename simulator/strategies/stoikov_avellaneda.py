from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from simulator.simulator import (MdUpdate, Order, OwnTrade, Sim,
                                 update_best_positions)


class StoikovAvellaneda():
    def __init__(
        self,
        delay: float,
        hold_time: Optional[float] = None,
        gamma: float = 0.5,
        sigma: float = 0.05,
        k: float = 0.5,
        time_trading_session_ends: float = 1655976252046013863,
        time_trading_session_starts: float = 1655942402250125991,
    ) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        self.gamma = gamma
        self.sigma = sigma
        self.time_trading_session_ends = time_trading_session_ends
        self.time_trading_session_starts = time_trading_session_starts
        self.k = k
    
    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf
        best_bid_vol = 0.000000001
        best_ask_vol = 0.000000001

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask, best_bid_vol, best_ask_vol = update_best_positions(
                        best_bid, best_ask, best_bid_vol, best_ask_vol, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #calculate reservation price
                # mid_price = (best_bid + best_ask) / 2
                mid_price = (best_bid + best_ask) / 2 
                q = len(ongoing_orders)
                sigma_2 = self.sigma**2
                time = (self.time_trading_session_ends - receive_ts) / \
                    (self.time_trading_session_ends - self.time_trading_session_starts)
                reservation_price = (
                    mid_price -
                    q * self.gamma *
                    sigma_2 * time
                )
                d_1 = self.gamma * sigma_2 * time
                d_2 = 2 / self.gamma * np.log(1 + self.gamma / self.k)
                delta = (d_1 + d_2) /2

                bid_price = reservation_price - delta
                ask_price = reservation_price + delta
                #place order
                bid_order = sim.place_order(
                    receive_ts, 0.001, 'BID', bid_price)
                ask_order = sim.place_order(
                    receive_ts, 0.001, 'ASK', ask_price)
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders
