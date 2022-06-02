import time

import TaxiEnv
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import numpy as np


class AgentGreedyImproved(AgentGreedy):
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        children, operators = get_children_and_operators(env, agent_id)
        children_heuristics = [heuristic_improved(child, agent_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]


def min_distance_from_passenger(env: TaxiEnv, position):
    passengers = env.passengers
    distance = [manhattan_distance(position, p.position) for p in passengers]
    return min(distance)


def min_distance_from_gas(env: TaxiEnv, position):
    gas_stations = env.gas_stations
    distance = [manhattan_distance(position, g.position) for g in gas_stations]
    return min(distance)


def cash_differece(env: TaxiEnv, taxi_id: int):
    taxi = env.get_taxi(taxi_id)
    other_taxi = env.get_taxi((taxi_id + 1) % 2)
    return taxi.cash - other_taxi.cash


def heuristic_improved(env: TaxiEnv, taxi_id: int):
    if env.done():
        if cash_differece(env, taxi_id) > 0:
            return np.inf
        elif cash_differece(env, taxi_id) == 0:
            return 0
        else:
            return -np.inf
    taxi = env.get_taxi(taxi_id)
    if taxi.passenger is not None:
        distance = manhattan_distance(taxi.position, taxi.passenger.destination) - 6
    else:
        distance = min_distance_from_passenger(env, taxi.position)
    distance_reward = -distance
    fuel_reward = taxi.fuel - distance
    gas_station_reward = min_distance_from_gas(env, taxi.position) / 2
    money_reward = 10 * cash_differece(env, taxi_id)
    env_huristic_value = distance_reward + fuel_reward + gas_station_reward + money_reward
    return env_huristic_value


def get_children_and_operators(env: TaxiEnv, agent_id):
    operators = env.get_legal_operators(agent_id)
    children = [env.clone() for _ in operators]
    for child, op in zip(children, operators):
        child.apply_operator(agent_id, op)
    return children, operators


class AgentMinimax(Agent):
    time_limit = 1
    start_time = 0

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        last_run_op = random.choice(operators)
        depth = 1
        while (depth <= env.num_steps):
            children, operators = get_children_and_operators(env, agent_id)
            children_heuristics = []
            out_of_time = False
            for child in children:
                child_heuristic = self.RB_minimax(child, agent_id, agent_id, depth - 1, depth - 1)
                if child_heuristic is None:
                    out_of_time = True
                    break
                children_heuristics.append(child_heuristic)
            if out_of_time:
                break
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
            last_run_op = operators[index_selected]
            depth += 1
        return last_run_op

    def RB_minimax(self, env: TaxiEnv, turn: int, agent_id: int, depth: int, original_depth: int):
        if (time.time() - self.start_time) > (self.time_limit - 0.01 * original_depth):
            return None
        if depth == 0 or env.done():
            return heuristic_improved(env, agent_id)
        turn = (turn + 1) % 2
        children, _ = get_children_and_operators(env, turn)
        if turn == agent_id:
            curr_max = -np.inf
            for child in children:
                child_minimax = self.RB_minimax(child, turn, agent_id, depth - 1, original_depth)
                if child_minimax is None:
                    return None
                curr_max = max(curr_max, child_minimax)
            return curr_max
        else:
            curr_min = np.inf
            for child in children:
                child_minimax = self.RB_minimax(child, turn, agent_id, depth - 1, original_depth)
                if child_minimax is None:
                    return None
                curr_min = min(curr_min, child_minimax)
            return curr_min


class AgentAlphaBeta(Agent):
    time_limit = 1
    start_time = 0

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        last_run_op = random.choice(operators)
        depth = 1
        while depth <= env.num_steps:
            children, operators = get_children_and_operators(env, agent_id)
            children_heuristics = []
            out_of_time = False
            alpha = -np.inf
            for child in children:
                child_heuristic = self.alpha_beta_minimax(child, agent_id, agent_id, depth - 1, depth - 1, alpha,
                                                          np.inf)
                if child_heuristic is None:
                    out_of_time = True
                    break
                alpha = max(alpha, child_heuristic)
                children_heuristics.append(child_heuristic)
            if out_of_time:
                break
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
            last_run_op = operators[index_selected]
            depth += 1
        return last_run_op

    def alpha_beta_minimax(self, env: TaxiEnv, turn: int, agent_id: int, depth: int, original_depth: int, alpha: int,
                           beta: int):
        if (time.time() - self.start_time) > (self.time_limit - 0.01 * original_depth):
            return None
        if depth == 0 or env.done():
            return heuristic_improved(env, agent_id)
        turn = (turn + 1) % 2
        children, _ = get_children_and_operators(env, turn)
        if turn == agent_id:
            curr_max = -np.inf
            for child in children:
                child_minimax = self.alpha_beta_minimax(child, turn, agent_id, depth - 1, original_depth, alpha, beta)
                if child_minimax is None:
                    return None
                curr_max = max(curr_max, child_minimax)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return np.inf
            return curr_max
        else:
            curr_min = np.inf
            for child in children:
                child_minimax = self.alpha_beta_minimax(child, turn, agent_id, depth - 1, original_depth, alpha, beta)
                if child_minimax is None:
                    return None
                curr_min = min(curr_min, child_minimax)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return -np.inf
            return curr_min


class AgentExpectimax(Agent):
    time_limit = 1
    start_time = 0

    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        self.time_limit = time_limit
        self.start_time = time.time()
        operators = env.get_legal_operators(agent_id)
        last_run_op = random.choice(operators)
        depth = 1
        while depth <= env.num_steps:
            children, operators = get_children_and_operators(env, agent_id)
            children_heuristics = []
            out_of_time = False
            for child in children:
                child_heuristic = self.expectimax(child, agent_id, agent_id, depth - 1, depth - 1)
                if child_heuristic is None:
                    out_of_time = True
                    break
                children_heuristics.append(child_heuristic)
            if out_of_time:
                break
            max_heuristic = max(children_heuristics)
            index_selected = children_heuristics.index(max_heuristic)
            last_run_op = operators[index_selected]
            depth += 1
        return last_run_op

    def get_states_probabilties(self, operators):
        special_operators = ["refuel", "drop off passenger", "pick up passenger", "park"]
        sum = 0
        operators_probability = {}
        for op in operators:
            if op in special_operators:
                operators_probability[op] = 2
                sum += 2
            else:
                operators_probability[op] = 1
                sum += 1
        for op in operators:
            operators_probability[op] /= sum
        return operators_probability

    def expectimax(self, env: TaxiEnv, turn: int, agent_id: int, depth: int, original_depth: int):
        if (time.time() - self.start_time) > (self.time_limit - 0.01 * original_depth):
            return None
        if depth == 0 or env.done():
            return heuristic_improved(env, agent_id)
        turn = (turn + 1) % 2
        children, operators = get_children_and_operators(env, turn)
        if turn == agent_id:
            curr_max = -np.inf
            for child in children:
                child_minimax = self.expectimax(child, turn, agent_id, depth - 1, original_depth)
                if child_minimax is None:
                    return None
                curr_max = max(curr_max, child_minimax)
            return curr_max
        else:
            expected_value = 0
            states_probabities = self.get_states_probabilties(operators)
            for child, op in zip(children, operators):
                child_expectimax = self.expectimax(child, turn, agent_id, depth - 1, original_depth)
                if child_expectimax is None:
                    return None
                expected_value += child_expectimax * states_probabities[op]
            return expected_value
