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
        children_heuristics = [self.heuristic_improved(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def min_distance_from_passenger(self, env: TaxiEnv, position):
        passengers = env.passengers
        distance = [manhattan_distance(position, p.position) for p in passengers ]
        print("distance: ")
        print(distance)
        return min(distance)


    def min_distance_from_gas(self, env: TaxiEnv, position):
        gas_stations = env.gas_stations
        distance = [manhattan_distance(position, g.position) for g in gas_stations]
        return min(distance)

    def heuristic_improved(self, env: TaxiEnv, taxi_id: int):
        taxi = env.get_taxi(taxi_id)
        if taxi.passenger is not None:
            distance = manhattan_distance(taxi.position, taxi.passenger.destination) - 6
        else:
            distance = self.min_distance_from_passenger(env, taxi.position)
        distance_reward =  -distance
        fuel_reward = taxi.fuel - distance
        gas_station_reward = self.min_distance_from_gas(env, taxi.position)/2
        money_reward = 10*self.heuristic(env, taxi_id)
        value = distance_reward + fuel_reward + gas_station_reward + money_reward
        return value


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
