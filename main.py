import random
import time
import numpy as np

from TaxiEnv import TaxiEnv
import argparse
import submission
import Agent
import itertools


def run_agents():
    parser = argparse.ArgumentParser(description='Test your submission by pitting agents against each other.')
    parser.add_argument('agent0', type=str,
                        help='First agent')
    parser.add_argument('agent1', type=str,
                        help='Second agent')
    parser.add_argument('-t', '--time_limit', type=float, nargs='?', help='Time limit for each turn in seconds', default=1)
    parser.add_argument('-s', '--seed', nargs='?', type=int, help='Seed to be used for generating the game',
                        default=random.randint(0, 255))
    parser.add_argument('-c', '--count_steps', nargs='?', type=int, help='Number of steps each taxi gets before game is over',
                        default=4761)
    parser.add_argument('--print_game', action='store_true')

    args = parser.parse_args()

    agents = {
        "random": Agent.AgentRandom(),
        "greedy": Agent.AgentGreedy(),
        "greedy_improved": submission.AgentGreedyImproved(),
        "minimax": submission.AgentMinimax(),
        "alphabeta": submission.AgentAlphaBeta(),
        "expectimax": submission.AgentExpectimax()
    }

    # agent_names = sys.argv
    agent_names = [args.agent0, args.agent1]
    env = TaxiEnv()

    env.generate(args.seed, 2*args.count_steps)

    if args.print_game:
        print('initial board:')
        env.print()

    for _ in range(args.count_steps):
        for i, agent_name in enumerate(agent_names):
            agent = agents[agent_name]
            start = time.time()
            op = agent.run_step(env, i, args.time_limit)
            end = time.time()
            if end - start > args.time_limit:
                raise RuntimeError("Agent used too much time!")
            env.apply_operator(i, op)
            if args.print_game:
                print('taxi ' + str(i) + ' chose ' + op)
                env.print()
        if env.done():
            break
    balances = env.get_balances()
    print(balances)
    if balances[0] == balances[1]:
        print('draw')
    else:
        print('taxi', balances.index(max(balances)), 'wins!')


def test(seed, count_steps ,agent_names):
    agents = {
        "random": Agent.AgentRandom(),
        "greedy": Agent.AgentGreedy(),
        "greedy_improved": submission.AgentGreedyImproved(),
        "minimax": submission.AgentMinimax(),
        "alphabeta": submission.AgentAlphaBeta(),
        "expectimax": submission.AgentExpectimax()
    }
    print_game = False
    time_limit = 10

    env = TaxiEnv()
    env.generate(seed, 2*count_steps)
    if print_game:
        print('initial board:')
        env.print()

    for _ in range(count_steps):
        for i, agent_name in enumerate(agent_names):
            agent = agents[agent_name]
            start = time.time()
            op = agent.run_step(env, i,time_limit)
            end = time.time()
            time_took = end - start
            #print("unused time: " + str(time_limit - time_took))

            if end - start > time_limit:
                raise RuntimeError("Agent used too much time!")
            env.apply_operator(i, op)
            if print_game:
                print('taxi ' + str(i) + ' chose ' + op)
                env.print()
        if env.done():
            break
    balances = env.get_balances()
    print(balances)
    if balances[0] == balances[1]:
        return 2
    else:
        return balances.index(max(balances))
    #else:
    #    print('taxi', balances.index(max(balances)), 'wins!')

def test_of_tests():
    results = [0, 0, 0]
    agent_names = [ "random", "greedy","greedy_improved","minimax", "alphabeta" ,"expectimax"]
    agents_wins = {
        "random": 0,
        "greedy": 0,
        "greedy_improved": 0,
        "minimax": 0,
        "alphabeta": 0,
        "expectimax": 0
    }
    draws_cnt = 0
    num_of_steps = 10
    for i in range(256):
        for agents_playing in itertools.product(agent_names, repeat = 2 ):
            if agents_playing[0] == agents_playing[1] or ("random" in agents_playing and "greedy" in agents_playing):
                continue
            print(agents_playing)
            #agents_playing = [agent1, agent2]
            result = test(i, num_of_steps, agents_playing)
            winner = "draw"
            if result < 2:
                agents_wins[agents_playing[result]] += 1
                winner = str(agents_playing[result])
            else:
                draws_cnt+=1
            print("seed: " + str(i) + ", players: " + str(agents_playing)+ ", winner: " + winner + ", draws: " + str(draws_cnt))
            print (agents_wins)

        #results[test(i, 10, agent_names)] += 1


    #results = [0, 0, 0]
    #agent_names = ["expectimax", "greedy_improved" ]
    #for i in range(10):
    #    results[test(i, 8, agent_names)] += 1
    #print("taxi 0 wins: " + str(results[0]))
    #print("taxi 1 wins: " + str(results[1]))
    #print("draws: " + str(results[2]))




if __name__ == "__main__":
    #test_of_tests()
    run_agents()

