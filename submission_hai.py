#from binarytree import Node
from Agent import Agent, AgentGreedy
from TaxiEnv import TaxiEnv, manhattan_distance
import random
import numpy as np
import math
import time

class AgentGreedyImproved(AgentGreedy):
    # TODO: section a : 3
    def run_step(self, env: TaxiEnv, taxi_id, time_limit):
        operators = env.get_legal_operators(taxi_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(taxi_id, op)
        children_heuristics = [self.heuristic(child, taxi_id) for child in children]
        max_heuristic = max(children_heuristics)
        index_selected = children_heuristics.index(max_heuristic)
        return operators[index_selected]

    def heuristic(self, env: TaxiEnv, taxi_id: int):
        reward = 0
        taxi_curr = env.get_taxi(taxi_id)
        taxi_rival = env.get_taxi(1-taxi_id)
        reward += (taxi_curr.fuel) * 5
        reward += (taxi_curr.cash - taxi_rival.cash) * 10
        penalties_gas = []
        for gas_station in env.gas_stations:
            destination = abs(gas_station.position[0] - taxi_curr.position[0]) + abs(gas_station.position[1] - taxi_curr.position[1]) 
            penalties_gas.append(destination)
        penalty_gas = min(penalties_gas)
        if not env.taxi_is_occupied(taxi_id):
            penalties = []
            for passenger in env.passengers:
                destination = abs(passenger.position[0] - taxi_curr.position[0]) + abs(passenger.position[1] - taxi_curr.position[1]) 
                penalties.append(destination)
            penalty_dest = min(penalties)
            if taxi_curr.fuel < penalty_gas + penalty_dest:
                return reward + (8 - penalty_gas)
            else:
                return reward + (8 - penalty_dest)
        else:
            passenger = taxi_curr.passenger
            penalty_dest = abs(passenger.destination[0] - taxi_curr.position[0]) + abs(passenger.destination[1] - taxi_curr.position[1]) 
            # now we need to check that resources > penalty
            if taxi_curr.fuel < penalty_gas + penalty_dest:
                return reward + (8 - penalty_gas)
            else:
                return reward + (8 - penalty_dest)


class AgentMinimax(Agent):

    def minimax_heuristic(self, env: TaxiEnv, taxi_id: int):
        reward = 0
        taxi_curr = env.get_taxi(taxi_id)
        taxi_rival = env.get_taxi(1-taxi_id)
        reward += (taxi_curr.cash) * 20
        penalties_gas = []
        for gas_station in env.gas_stations:
            destination = abs(gas_station.position[0] - taxi_curr.position[0]) + abs(gas_station.position[1] - taxi_curr.position[1]) 
            penalties_gas.append(destination)
        penalty_gas = min(penalties_gas)
        if not env.taxi_is_occupied(taxi_id):
            penalties = []
            for passenger in env.passengers:
                destination = abs(passenger.position[0] - taxi_curr.position[0]) + abs(passenger.position[1] - taxi_curr.position[1]) 
                penalties.append(destination)
            penalty_dest = min(penalties)
            if taxi_curr.fuel < penalty_dest:
                return reward + (8-penalty_gas)*3 
            else:
                return reward + (8-penalty_gas) + (8-penalty_dest)*8
        else:
            passenger = taxi_curr.passenger
            penalty_dest = abs(passenger.destination[0] - taxi_curr.position[0]) + abs(passenger.destination[1] - taxi_curr.position[1]) 
            if taxi_curr.fuel < penalty_dest:
                return reward + (8-penalty_gas)*3
            else:
                return reward + (8-penalty_gas) + (8-penalty_dest)*8

    def minimax(self, env: TaxiEnv, taxi_id_orig: int, taxi_id_curr: int, time_limit: int, depth: int):
        AgentMinimax.time_limit_global = time_limit
        AgentMinimax.num_nodes = 0
        AgentMinimax.finish = False
        def warapper(env: TaxiEnv, taxi_id_orig: int, taxi_id_curr: int, depth: int):
            start_time = time.time() 
            val = self.minimax_heuristic(env, taxi_id_orig)
            if env.done() or depth == 0:
                return val
            operators = env.get_legal_operators(taxi_id_curr)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(taxi_id_curr, op)
            epsilon =0
            if taxi_id_orig == taxi_id_curr:
                curr_max = -math.inf
                epsilon = time.time() - start_time  
                AgentMinimax.time_limit_global -= epsilon   
                if ((AgentMinimax.num_nodes*len(children))*(epsilon) >= AgentMinimax.time_limit_global - epsilon):
                    AgentMinimax.finish = True
                    return val
                for child in children:
                    AgentMinimax.num_nodes +=1
                    v = warapper(child, taxi_id_orig, abs(taxi_id_curr-1), depth-1)
                    curr_max = max(v, curr_max)
                    if AgentMinimax.finish:
                        break
                return curr_max
            else:
                curr_min = math.inf
                epsilon = time.time() - start_time  
                AgentMinimax.time_limit_global -= epsilon   
                if ((AgentMinimax.num_nodes*len(children))*(epsilon) >= AgentMinimax.time_limit_global - epsilon):
                    AgentMinimax.finish = True
                    return val
                for child in children:
                    AgentMinimax.num_nodes +=1
                    v = warapper(child, taxi_id_orig, abs(taxi_id_curr-1), depth-1)
                    curr_min = min(v, curr_min)
                    if AgentMinimax.finish:
                        break
                return curr_min
        return warapper(env, taxi_id_orig, taxi_id_curr, depth)
        
    
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        epsilon_run_step = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        len_children = len(children)
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        start_curr_depth = time.time() - epsilon_run_step
        children_heuristics = [self.minimax(child, agent_id, agent_id, (time_limit - start_curr_depth)/len_children, 0) for child in children]
        best_op_val = max(children_heuristics)
        best_op_idx = children_heuristics.index(best_op_val)
        for depth in range(1,100,2):
            if AgentMinimax.finish == True:
                break
            start_curr_depth = time.time() - epsilon_run_step
            children_heuristics = [self.minimax(child, agent_id, agent_id, (time_limit - start_curr_depth)/len_children, depth) for child in children]
            max_heuristic = max(children_heuristics)
            if(max_heuristic > best_op_val):
                best_op_val = max_heuristic
                best_op_idx = children_heuristics.index(max_heuristic)
        return operators[best_op_idx]
        


class AgentAlphaBeta(Agent):

    def alpha_beta_heuristic(self, env: TaxiEnv, taxi_id: int):
        reward = 0
        taxi_curr = env.get_taxi(taxi_id)
        taxi_rival = env.get_taxi(1-taxi_id)
        reward += (taxi_curr.cash) * 20
        penalties_gas = []
        for gas_station in env.gas_stations:
            destination = abs(gas_station.position[0] - taxi_curr.position[0]) + abs(gas_station.position[1] - taxi_curr.position[1]) 
            penalties_gas.append(destination)
        penalty_gas = min(penalties_gas)
        if not env.taxi_is_occupied(taxi_id):
            penalties = []
            for passenger in env.passengers:
                destination = abs(passenger.position[0] - taxi_curr.position[0]) + abs(passenger.position[1] - taxi_curr.position[1]) 
                penalties.append(destination)
            penalty_dest = min(penalties)
            if taxi_curr.fuel < penalty_dest:
                return reward + (8-penalty_gas)*3 
            else:
                return reward + (8-penalty_gas) + (8-penalty_dest)*8
        else:
            passenger = taxi_curr.passenger
            penalty_dest = abs(passenger.destination[0] - taxi_curr.position[0]) + abs(passenger.destination[1] - taxi_curr.position[1]) 
            if taxi_curr.fuel < penalty_dest:
                return reward + (8-penalty_gas)*3
            else:
                return reward + (8-penalty_gas) + (8-penalty_dest)*8

    def minimax(self, env: TaxiEnv, taxi_id_orig: int, taxi_id_curr: int, time_limit: int, depth: int, alpha: int, beta: int):
        AgentMinimax.time_limit_global = time_limit
        AgentMinimax.num_nodes = 0
        AgentMinimax.finish = False
        def warapper(env: TaxiEnv, taxi_id_orig: int, taxi_id_curr: int, depth: int, alpha: int, beta: int):
            start_time = time.time() 
            val = self.alpha_beta_heuristic(env, taxi_id_orig)
            if env.done() or depth == 0:
                return val
            operators = env.get_legal_operators(taxi_id_curr)
            children = [env.clone() for _ in operators]
            for child, op in zip(children, operators):
                child.apply_operator(taxi_id_curr, op)
            epsilon =0
            if taxi_id_orig == taxi_id_curr:
                curr_max = -math.inf
                epsilon = time.time() - start_time  
                AgentMinimax.time_limit_global -= epsilon   
                if ((AgentMinimax.num_nodes*len(children))*(epsilon) >= AgentMinimax.time_limit_global - epsilon):
                    AgentMinimax.finish = True
                    return val
                for child in children:
                    AgentMinimax.num_nodes +=1
                    v = warapper(child, taxi_id_orig, abs(taxi_id_curr-1), depth-1, alpha, beta)
                    curr_max = max(v, curr_max)
                    alpha = max(curr_max, alpha)
                    if curr_max >= beta:
                        return math.inf
                    if AgentMinimax.finish:
                        break
                return curr_max
            else:
                curr_min = math.inf
                epsilon = time.time() - start_time  
                AgentMinimax.time_limit_global -= epsilon   
                if ((AgentMinimax.num_nodes*len(children))*(epsilon) >= AgentMinimax.time_limit_global - epsilon):
                    AgentMinimax.finish = True
                    return val
                for child in children:
                    AgentMinimax.num_nodes +=1
                    v = warapper(child, taxi_id_orig, abs(taxi_id_curr-1), depth-1, alpha, beta)
                    curr_min = min(v, curr_min)
                    beta = min(curr_min, beta)
                    if curr_min <= alpha:
                        return -math.inf
                    if AgentMinimax.finish:
                        break
                return curr_min
        return warapper(env, taxi_id_orig, taxi_id_curr, depth, alpha, beta)
        
    
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        epsilon_run_step = time.time()
        operators = env.get_legal_operators(agent_id)
        children = [env.clone() for _ in operators]
        len_children = len(children)
        for child, op in zip(children, operators):
            child.apply_operator(agent_id, op)
        start_curr_depth = time.time() - epsilon_run_step
        children_heuristics = [self.minimax(child, agent_id, agent_id, (time_limit - start_curr_depth)/len_children, 0, -math.inf, math.inf) for child in children]
        best_op_val = max(children_heuristics)
        best_op_idx = children_heuristics.index(best_op_val)
        for depth in range(1,100,2):
            if AgentMinimax.finish == True:
                break
            start_curr_depth = time.time() - epsilon_run_step
            children_heuristics = [self.minimax(child, agent_id, agent_id, (time_limit - start_curr_depth)/len_children, depth, -math.inf, math.inf) for child in children]
            max_heuristic = max(children_heuristics)
            if(max_heuristic > best_op_val):
                best_op_val = max_heuristic
                best_op_idx = children_heuristics.index(max_heuristic)
        return operators[best_op_idx]


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: TaxiEnv, agent_id, time_limit):
        raise NotImplementedError()
