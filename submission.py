from datetime import time

import TaxiEnv
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random


class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        children_heuristics = [heuristic_improved(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

def min_distance_from_passenger(env: TaxiEnv, position):
    passengers = env.passengers
    distance = [manhattan_distance(position, p.position) for p in passengers ]
    return min(distance)


def min_distance_from_gas (env: TaxiEnv, position):
    gas_stations = env.gas_stations
    distance = [manhattan_distance(position, g.position) for g in gas_stations]
    return min(distance)

def cash_differece(env: TaxiEnv, taxi_id: int):
    taxi = env.get_taxi(taxi_id)
    other_taxi = env.get_taxi((taxi_id + 1) % 2)
    return taxi.cash - other_taxi.cash

#TODO: should add extreme vale when env.done
def heuristic_improved( env: TaxiEnv, taxi_id: int):
    taxi = env.get_taxi(taxi_id)
    if taxi.passenger is not None:
        distance = manhattan_distance(taxi.position, taxi.passenger.destination) - 6
    else:
        distance = min_distance_from_passenger(env, taxi.position)
    distance_reward =  -distance
    fuel_reward = taxi.fuel - distance
    gas_station_reward = min_distance_from_gas(env, taxi.position)/2
    money_reward = 10*cash_differece(env, taxi_id)
    env_huristic_value = distance_reward + fuel_reward + gas_station_reward + money_reward
    return env_huristic_value




class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        start = time.time()
        time_for_run_step_calc = time_limit*0.8
        operators = env.get_legal_operators(agent_id)
        last_run_op = random.choice(operators)
        depth = 0
        while time.time() - start  < time_for_run_step_calc or depth > env.num_steps:
            #TODO: this can still run for a long time if the fuel is over maybe need to change?
            operators = env.get_legal_operators(agent_id)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(agent_id, op)
            children_heuristics = [self.RB_minimax(child, (agent_id+1)%2, depth) for child in children]
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
            last_run_op =  operators[index_selected]
            depth += 1
        return last_run_op




    def RB_minimax(self, env: TaxiEnv, agent_id, depth: int):
        if env.done():
            return env.get_balances()[agent_id]
        if depth == 0:
            return heuristic_improved(env, agent_id)


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
