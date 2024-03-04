import numpy as np
import random, time
import itertools
from automata_learning_utils import al_utils
# import automata_learning_utils.al_utils
from worlds.game import *
from reward_machines.reward_machine import RewardMachine
from automata_learning.Traces import Traces
from tester.tester import Tester
from tester.livetester import LiveTester
from tester.timer import Timer
import shutil
import os
import subprocess
import csv
import time
import openai
from dotenv import load_dotenv
import json
import LLM_lib.LLM_utils as llm_utils
import copy




def run_aqrm_task(epsilon, env, learned_rm_file, 
                  tester_true, tester_learned,
                    curriculum, show_print,
                      is_rm_learned, currentstep,
                        previous_testing_reward, q, testing_params):
    """
    This code runs one training episode.
        - rm_file: It is the path towards the RM machine to solve on this episode
        - environment_rm: an environment reward machine, the "true" one, underlying the execution
    """
    print('learned_rm_file:', learned_rm_file)
    print('tester_true:', tester_true)

    # exit()
    # Initializing parameters and the game
    learning_params = tester_learned.learning_params
    testing_params = tester_learned.testing_params

    print('agents num in run_aqrm_task is:', testing_params.agents_num)
    # exit()
    """
     here, tester holds all the machines. we would like to dynamically update the machines every so often.
     an option might be to read it every time a new machine is learnt
     """
    reward_machines = [tester_learned.get_hypothesis_machine()]

    task_params = tester_learned.get_task_params(learned_rm_file) # rm_files redundant here unless in water world (in which case it provides the map files based on the task)
    rm_true = tester_true.get_reward_machines()[0] # add one more input n to track tasks at hand, replace 0 with n
    rm_learned = tester_learned.get_hypothesis_machine()

    task = Game(task_params)


    # exit()
    actions = task.get_actions()
    num_features = len(task.get_features())
    num_steps = learning_params.max_timesteps_per_task
    training_reward = 0
    is_conflicting = 1 #by default add traces
    testing_reward = None #initialize

    # Getting the initial state of the environment and the reward machine
    s1_list = [[] for _ in range(testing_params.agents_num)]
    s1_features_list = [[] for _ in range(testing_params.agents_num)]
    u1_list = [[] for _ in range(testing_params.agents_num)]
    u1_true_list = [[] for _ in range(testing_params.agents_num)]
    s_x, s_y = [[] for _ in range(testing_params.agents_num)], [[] for _ in range(testing_params.agents_num)]

    for agent_ind in range(testing_params.agents_num):
        s1_list[agent_ind], features_temp =\
              task.get_state_and_features()
        s1_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
        # s_x[agent_ind], s_y[agent_ind] = task.get_state_vector(s1_features_list[agent_ind])
        s_x[agent_ind], s_y[agent_ind] = s_x_temp, s_y_temp
        print('s_x is:', s_x[agent_ind])
        print('s_y is:', s_y[agent_ind])



        u1_list[agent_ind] = rm_learned.get_initial_state()
        u1_true_list[agent_ind] = rm_true.get_initial_state()
    
    print('s_y list is:', s_y)
    print('s_x list is:', s_x)

    # exit()

    has_been = [0,0]
    alpha = 0.9
    gamma = 0.9
    w = 0
    T = 100

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps)
    all_events = []
    sy_s = [[]]
    a_s = []
    a=0
    l = 0
    c = 0
    p = 0

    reward_list = [0 for _ in range(testing_params.agents_num)]

    for t in range(num_steps):
        currentstep += 1
        s_list = [[] for _ in range(testing_params.agents_num)]

        for agent_ind in range(testing_params.agents_num):
            s_list[agent_ind] = np.where(s1_features_list[agent_ind]==1)[0][0]


        # State update for light, pedestrian, car
        # if t//10:
        #     l = np.random.randint(0, 2)
        #     c = np.random.randint(0, 2)
        #     p = np.random.randint(0, 2)

        print('l', l, 'c', c, 'p', p)
        # Choosing an action to perform
        # if random.random() < epsilon:
        # if random.random() < 0.15:

        actions_list = [[] for _ in range(testing_params.agents_num)]
        for agent_ind in range(testing_params.agents_num):
            
            
            if random.random() < 0.30:
                a = random.choice(actions)
            else:
                if max(q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]])==0:
                    a = random.choice(actions)
                else:
                    a = np.argmax(q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]])
            print('actions is', a)
            actions_list[agent_ind] = a
            
        print('action list is:', actions_list)
        print('q of agents dim is:', q.shape)

        
        # exit()
        # updating the curriculum
        curriculum.add_step()

        s2_list = [[] for _ in range(testing_params.agents_num)]
        s2_features_list = [[] for _ in range(testing_params.agents_num)] 
        s_new_list = [[] for _ in range(testing_params.agents_num)]   

        # Executing the action
        events_list = [[] for _ in range(testing_params.agents_num)]
        if tester_learned.game_type=="trafficworld":
            for agent_ind in range(testing_params.agents_num):
                events_list[agent_ind] = task.get_true_propositions_action(
                    actions_list[agent_ind])
                task.execute_action(s_x[agent_ind], s_y[agent_ind],actions_list[agent_ind],l,c,p)
                actions_list[agent_ind] = task.get_last_action() # due to MDP slip
        else:
            for agent_ind in range(testing_params.agents_num):
                print('before execute action in aqrm task')

                print('s_x before exec action:', s_x[agent_ind])
                print('s_y before exec action:', s_y[agent_ind])
                print('s_x list before exec action:', s_x)
                print('s_y list before exec action:', s_y)

                # exit()
                s_x[agent_ind], s_y[agent_ind] = \
                    task.execute_action(s_x[agent_ind], s_y[agent_ind], actions_list[agent_ind],l,c,p)
                print('s_x_temp is:', s_x_temp)

                print('after execute action in aqrm task')
                # actions_list[agent_ind] = task.get_last_action() # due to MDP slip
                events_list[agent_ind] = task.get_true_propositions()
                # s2_list[agent_ind], s2_features_list[agent_ind] =\
                #       task.get_state_and_features()
                s2_list[agent_ind], features_temp =\
                                task.get_state_and_features()
                s2_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
                # s_x[agent_ind], s_y[agent_ind] = task.get_state_vector(s1_features_list[agent_ind])
                s_x[agent_ind], s_y[agent_ind] = s_x_temp, s_y_temp
                # print('after get feature in aqrm task s2 feature list is:', s2_features_list[agent_ind], 'agent is:', agent_ind)
                s_new_list[agent_ind] = np.where(s2_features_list[agent_ind]==1)[0][0]
                # print('s_new_list for agent is:', s_new_list[agent_ind])
                # print('s_new_list is:', s_new_list)
        # exit()
        ###############################################
        # print(">>>", a)
        # # print(task.game)
        # task.game.show()
        # # time.sleep(0.2)
        # print(">>>", events)
        u2_list = [[] for _ in range(testing_params.agents_num)]
        u2_true_list = [[] for _ in range(testing_params.agents_num)]
        # reward_list = [[] for _ in range(testing_params.agents_num)]
        for agent_ind in range(testing_params.agents_num):
            
            u2_list[agent_ind] = rm_learned.get_next_state(u1_list[agent_ind],
                                            events_list[agent_ind])
            
            u2_true_list[agent_ind] = rm_true.get_next_state(u1_true_list[agent_ind],
                                                             events_list[agent_ind])
            
            # reward_list[agent_ind] = rm_true.get_reward(u1_true_list[agent_ind]
            #                                             ,u2_true_list[agent_ind]
            #                                             ,s1_list[agent_ind],
            #                                             actions_list[agent_ind],
            #                                             s2_list[agent_ind])
            if  reward_list[agent_ind]<=0:
                # reward[i] = swarm_reward_machine.get_reward(u1_swarm[i],u2_swarm[i],s[i],a[i],s_new[i])
                reward_list[agent_ind] = rm_true.get_reward(u1_true_list[agent_ind]
                                                        ,u2_true_list[agent_ind]
                                                        ,s1_list[agent_ind],
                                                        actions_list[agent_ind],
                                                        s2_list[agent_ind])

        for agent_ind in range(testing_params.agents_num):
            q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]][actions_list[agent_ind]] =\
                  (1 - alpha)\
                  * q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]][actions_list[agent_ind]] + \
                    alpha * (reward_list[agent_ind] + gamma * np.amax(q[agent_ind][s_new_list[agent_ind]][u2_list[agent_ind]]))

        # exit()
        # sy = s%9
        # sx = (s-sy)/9
        # synew = s_new % 9
        # sxnew = (s_new - synew) / 9
        # a1=a

        # for agent_ind in range(testing_params.agents_num):

        all_events.append(events_list[agent_ind])

        # training_reward += reward
        training_reward += np.mean(reward_list)
        print('reward of agent No.1:', reward_list[0])
        print('reward of agent No.2:', reward_list[1])
        print('reward of agent No.3:', reward_list[2])
        print('reward of agent No.4:', reward_list[3])



        # Getting rewards and next states for each reward machine
        rewards, next_states = [],[]
        rewards_hyp, next_states_hyp = [],[]
        j_rewards, j_next_states = rm_true.get_rewards_and_next_states(s1_list[agent_ind], 
                            actions_list[agent_ind], s2_list[agent_ind], events_list[agent_ind])
        rewards.append(j_rewards)
        next_states.append(j_next_states)

        j_rewards_hyp, j_next_states_hyp = rm_learned.get_rewards_and_next_states(
            s1_list[agent_ind],
                            actions_list[agent_ind], s2_list[agent_ind],
                              events_list[agent_ind])
        rewards_hyp.append(j_rewards_hyp)
        next_states_hyp.append(j_next_states_hyp)



        # Printing
        if show_print and (t+1) % learning_params.print_freq == 0:
            print("Step:", t+1, "\tTotal reward:", training_reward)

        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq==0:
            testing_reward = tester_learned.run_test(curriculum.get_current_step(), run_aqrm_test, rm_learned, rm_true, is_rm_learned, q, num_features)


        if is_rm_learned==0:
            # for agent_ind in range(testing_params.agents_num):
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_true_list[agent_ind]):
                # Restarting the game
                task = Game(task_params)
                if curriculum.stop_task(t):
                    break

                for agent_ind in range(testing_params.agents_num):
                    # s2_list[agent_ind], s2_features_list[agent_ind], s_x_temp, s_y_temp =\
                    #     task.get_state_and_features()
                    s2_list[agent_ind], features_temp =\
                                task.get_state_and_features()
                    s2_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
                    u2_true_list[agent_ind] = rm_true.get_initial_state()


        else:
            # for agent_ind in range(testing_params.agents_num):
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_true_list[agent_ind]):
                # Restarting the game
                task = Game(task_params)

                if curriculum.stop_task(t):
                    break

                for agent_ind in range(testing_params.agents_num):
                    # s2_list[agent_ind], s2_features_list[agent_ind], s_x_temp, s_y_temp =\
                    #         task.get_state_and_features()
                    s2_list[agent_ind], features_temp =\
                                task.get_state_and_features()
                    s2_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
                    u2_true_list[agent_ind] = rm_true.get_initial_state()
                    u2_list[agent_ind] = rm_learned.get_initial_state()


        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        for agent_ind in range(testing_params.agents_num):
            s1_list[agent_ind], s1_features_list[agent_ind], u1_list[agent_ind]\
                  = copy.deepcopy(s2_list[agent_ind]),\
                      copy.deepcopy(s2_features_list[agent_ind]),\
                          copy.deepcopy(u2_list[agent_ind])
            u1_true_list[agent_ind] = copy.deepcopy(u2_true_list[agent_ind])

    # for agent_ind in range(testing_params.agents_num):
    if rm_true.is_terminal_state(u2_true_list[agent_ind]):
        checker = rm_learned.is_terminal_state(u2_list[agent_ind])
    # (is_rm_learned) and
    if (not rm_learned.is_terminal_state(u2_list[agent_ind])) and (not rm_true.is_terminal_state(u2_true_list[agent_ind])):
        is_conflicting = 0
    elif (rm_learned.is_terminal_state(u2_list[agent_ind]) and rm_true.is_terminal_state(u2_true_list[agent_ind])):
        is_conflicting = 0
    else:
        is_conflicting = 1

    step_count=t

    if testing_reward is None:
        is_test_result = 0
        testing_reward = previous_testing_reward
    else:
        is_test_result = 1

    if show_print: print("Done! Total reward:", training_reward)

    return all_events, training_reward, step_count, is_conflicting, testing_reward, is_test_result, q


def run_aqrm_test(reward_machines, task_params, rm, rm_true, is_learned, q, learning_params, testing_params, optimal, num_features):
    # Initializing parameters
    task = Game(task_params)

    
    s1_list = [[] for _ in range(testing_params.agents_num)]
    s1_features_list = [[] for _ in range(testing_params.agents_num)]
    u1_list = [[] for _ in range(testing_params.agents_num)]
    u1_true_list = [[] for _ in range(testing_params.agents_num)]
    s_x, s_y = [[] for _ in range(testing_params.agents_num)], [[] for _ in range(testing_params.agents_num)]

    for agent_ind in range(testing_params.agents_num):
        s1_list[agent_ind], s1_features_list[agent_ind] =\
              task.get_state_and_features()
        
        s1_list[agent_ind], features_temp =\
              task.get_state_and_features()
        s1_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
        # s_x[agent_ind], s_y[agent_ind] = task.get_state_vector(s1_features_list[agent_ind])
        s_x[agent_ind], s_y[agent_ind] = s_x_temp, s_y_temp

        u1_list[agent_ind] = rm.get_initial_state()
        u1_true_list[agent_ind] = rm_true.get_initial_state()
    
    # s1, s1_features = task.get_state_and_features()

    # u1 = rm.get_initial_state()
    # u1_true = rm_true.get_initial_state()

    alpha = 0.9
    gamma = 0.9
    w = 0
    ok = 0
    T = 100

    # Starting interaction with the environment
    r_total = 0

    l = 0
    c = 0
    p = 0

    r_list = [0 for _ in range(testing_params.agents_num)]

    for t in range(testing_params.num_steps):

        # Choosing an action to perform
        actions = task.get_actions()
        # s = np.where(s1_features==1)[0][0]

        s_list = [[] for _ in range(testing_params.agents_num)]

        for agent_ind in range(testing_params.agents_num):
            s_list[agent_ind] = np.where(s1_features_list[agent_ind]==1)[0][0]


        # State update for light, pedestrian, car
        # if t//10:
        #     l = np.random.randint(0, 2)
        #     c = np.random.randint(0, 2)
        #     p = np.random.randint(0, 2)

        actions_list = [[] for _ in range(testing_params.agents_num)]
        for agent_ind in range(testing_params.agents_num):
            if max(q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]]) == 0:
                a = random.choice(actions)
            else:
                a = np.argmax(q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]])

            actions_list[agent_ind] = a

        # Executing the action
        events_list = [[] for _ in range(testing_params.agents_num)]
        s2_list = [[] for _ in range(testing_params.agents_num)]
        s2_features_list = [[] for _ in range(testing_params.agents_num)] 
        s_new_list = [[] for _ in range(testing_params.agents_num)]  
        u2_list = [[] for _ in range(testing_params.agents_num)]
        u2_true_list = [[] for _ in range(testing_params.agents_num)]
        # r_list = [0 for _ in range(testing_params.agents_num)]
        print('r list initial is:', r_list)
        # exit()

        if task_params.game_type=="trafficworld":
            for agent_ind in range(testing_params.agents_num):
                events_list[agent_ind] = task.get_true_propositions_action(actions_list[agent_ind])
                task.execute_action(actions_list[agent_ind],l,c,p)
                actions_list[agent_ind] = task.get_last_action() # due to MDP slip
        else:
            for agent_ind in range(testing_params.agents_num):
                # task.execute_action(actions_list[agent_ind],l,c,p)
                s_x[agent_ind], s_y[agent_ind] = \
                    task.execute_action(s_x[agent_ind], s_y[agent_ind], actions_list[agent_ind],l,c,p)
                actions_list[agent_ind] = task.get_last_action() # due to MDP slip
                events_list[agent_ind] = task.get_true_propositions()



                # s_x[agent_ind], s_y[agent_ind] = \
                #     task.execute_action(s_x[agent_ind], s_y[agent_ind], actions_list[agent_ind],l,c,p)
                # print('s_x_temp is:', s_x_temp)

                # print('after execute action in aqrm task')
                # # actions_list[agent_ind] = task.get_last_action() # due to MDP slip
                # events_list[agent_ind] = task.get_true_propositions()
                # # s2_list[agent_ind], s2_features_list[agent_ind] =\
                # #       task.get_state_and_features()
                # s2_list[agent_ind], features_temp =\
                #                 task.get_state_and_features()
                # s2_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
                # # s_x[agent_ind], s_y[agent_ind] = task.get_state_vector(s1_features_list[agent_ind])
                # s_x[agent_ind], s_y[agent_ind] = s_x_temp, s_y_temp


        for agent_ind in range(testing_params.agents_num):
            # s2_list[agent_ind], s2_features_list[agent_ind] =\
            #       task.get_state_and_features()
            s2_list[agent_ind], features_temp =\
                            task.get_state_and_features()
            s2_features_list[agent_ind], s_x_temp, s_y_temp = features_temp
            
            u2_list[agent_ind] = rm.get_next_state(u1_list[agent_ind], 
                                                   events_list[agent_ind])
            u2_true_list[agent_ind] = rm_true.get_next_state(u1_true_list[agent_ind],
                                                              events_list[agent_ind])
            # r_list[agent_ind] = rm_true.get_reward(u1_true_list[agent_ind],
            #                                        u2_true_list[agent_ind],
            #                                        s1_list[agent_ind],
            #                                        actions_list[agent_ind],
            #                                        s2_list[agent_ind])
            
            if  r_list[agent_ind]<=0:
                # reward[i] = swarm_reward_machine.get_reward(u1_swarm[i],u2_swarm[i],s[i],a[i],s_new[i])
                r_list[agent_ind] = rm_true.get_reward(u1_true_list[agent_ind],
                                                   u2_true_list[agent_ind],
                                                   s1_list[agent_ind],
                                                   actions_list[agent_ind],
                                                   s2_list[agent_ind])
            
            
            print('agent ind is:', agent_ind)
            print('u1_list is:', u1_list[agent_ind])
            print('u2_list is:', u2_list[agent_ind])

            print('u2_true_list is:', u2_true_list[agent_ind])
            print('reward is:', r_list[agent_ind])



            s_new_list[agent_ind] = np.where(s2_features_list[agent_ind]==1)[0][0]

            q[agent_ind][s_list[agent_ind]][u1_list[agent_ind]][actions_list[agent_ind]] =\
                  (1 - alpha) * q[agent_ind][s_list[agent_ind]]\
                    [u1_list[agent_ind]][actions_list[agent_ind]] + \
                alpha * (r_list[agent_ind] + gamma *\
                          np.amax(q[agent_ind][s_new_list[agent_ind]][u2_list[agent_ind]]))


        # r_total += r * learning_params.gamma**t # used in original graphing framework
        r_total += np.mean(r_list) * learning_params.gamma**t # used in original graphing framework

        # Restarting the environment (Game Over)
        print('u2_true_list is:', u2_true_list)
        print('u2_true terminal:', rm_true.is_terminal_state(u2_true_list[0]))
        
        u_terminal_list = [rm_true.is_terminal_state(u2_true_list[agent_ind]) 
                           for agent_ind in range(testing_params.agents_num) ]
        print('u_terminal_list is:', u_terminal_list)
        if is_learned==0:
            # for agent_ind in range(testing_params.agents_num):
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_true_list[0]):
            # if task.is_env_game_over() or all(u_terminal_list):
                # for agent_ind in range(testing_params.agents_num):
                #     r_list[agent_ind] = 1
                break

        else:
            # for agent_ind in range(testing_params.agents_num):
            if task.is_env_game_over() or rm_true.is_terminal_state(u2_true_list[0]):
            # if task.is_env_game_over() or all(u_terminal_list):
            #     for agent_ind in range(testing_params.agents_num):
            #         r_list[agent_ind] = 1
                break

        # Moving to the next state
        for agent_ind in range(testing_params.agents_num):
            s1_list[agent_ind], s1_features_list[agent_ind], u1_list[agent_ind]\
                  = copy.deepcopy(s2_list[agent_ind]),\
                    copy.deepcopy(s2_features_list[agent_ind]),\
                    copy.deepcopy(u2_list[agent_ind])
            u1_true_list[agent_ind] = copy.deepcopy(u2_true_list[agent_ind])

    reward_sum = np.mean(r_list[0])
    print('reward test of agent No.1:', r_list[0])
    print('reward test of agent No.2:', r_list[1])
    print('reward test of agent No.3:', r_list[2])
    print('reward test of agent No.4:', r_list[3])
    # for agent_ind in range(testing_params.agents_num):
    # if rm_true.is_terminal_state(u2_true_list[agent_ind]) and r_list[agent_ind]>0:
    #     return 1
    # else:
    #     return 0

    return reward_sum

def _remove_files_from_folder(relative_path):

    dirname = os.path.abspath(os.path.dirname(__file__))


    parent_folder = os.path.normpath(os.path.join(dirname, relative_path))

    if os.path.isdir(parent_folder):
        for filename in os.listdir(parent_folder):
            absPath = os.path.join(parent_folder, filename)
            subprocess.run(["rm", absPath])
    else:
        print("There is no directory {}".format(parent_folder))


def run_aqrm_experiments(alg_name, testing_params, tester, tester_learned, curriculum, num_times,
                          show_print, show_plots, al_alg_name, sat_alg_name, 
                          pysat_hints=None, prompt_file=None, messages=None):
    alg_name = alg_name.lower()
    al_alg_name = al_alg_name.lower()
    sat_alg_name = sat_alg_name.lower()

    testing_params = tester_learned.testing_params
    learning_params = tester_learned.learning_params

    # print('agents num is:', testing_params.agents_num)
    # exit()

    algorithm_name = alg_name
    if alg_name=="jirp":
        algorithm_name += al_alg_name
        if al_alg_name=="pysat":
            algorithm_name += sat_alg_name
        if pysat_hints is not None:
            algorithm_name += ":" + ":".join(hint.lower() for hint in pysat_hints)
    print('tester world task', tester.world.tasks[0])
    print('pysat hint is:', pysat_hints)
    print('prompt in run_aqrm:', prompt_file)
    # exit()
    for character in tester.world.tasks[0]:
        if str.isdigit(character):
            task_id = character
            print('task id', task_id)
    
    run_name = f"{tester.game_type}{task_id}{algorithm_name}"
    print('run name', run_name)
    # exit()
    details_filename = f"../plotdata/details_{run_name}.csv"
    with open(details_filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["RUN_PARAMETERS:"])
        wr.writerow(["world:",         tester.game_type])
        wr.writerow(["  task:",        task_id])
        wr.writerow(["algorithm:",     alg_name])
        wr.writerow(["  enter_loop:",  learning_params.enter_loop])
        wr.writerow(["  step_unit:",   testing_params.num_steps])
        wr.writerow(["  total_steps:", curriculum.total_steps])
        wr.writerow(["  epsilon:",     0.30]) # WARNING: it's hard-coded in `run_aqrm_task()`
        wr.writerow(["al_algorithm:", al_alg_name.upper()])
        if "PYSAT" in al_alg_name.upper():
            wr.writerow(["  sat_algorithm:", sat_alg_name])
            if pysat_hints is None:
                wr.writerow(["  hint_at:", "never"])
            else:
                wr.writerow(["  hint_at:", "relearn"]) # WARNING: it's hard-coded (see `hm_file`)
                for hint in pysat_hints:
                    wr.writerow(["    hint:", hint.lower() if hint else "∅"])



    # time_start = time.clock()

    # just in case, delete all temporary files
    dirname = os.path.abspath(os.path.dirname(__file__))
    _remove_files_from_folder("../automata_learning_utils/data")
    # exit()
    # Running the tasks 'num_times'
    time_init = time.time()
    plot_dict = dict()
    rewards_plot = list()

    # LLM call to creat the hint DFA
    llm_dfa = llm_utils.LLM_DFA()

    print('message is:', messages)
    # exit()
    print('prompt file:', prompt_file[0])
    prompt_obj = open(prompt_file[0])
    prompt_data = json.load(prompt_obj)
    instruct_for_gpt = prompt_data['instructions'][0]['comment']

    messages.append({'role': 'system', 'content': "You are an expert in Deterministic Finite Automata in the field of computer science"})
    messages.append({'role': 'user', 'content': instruct_for_gpt})

    # prompt_file = [{'role':'user', 'content': message}]
    prompt = messages

    params = llm_dfa.set_open_params()
    gpt_output = llm_dfa.get_completion(params, prompt)
    response = gpt_output.choices[0].message.content
    print('response in aqrm:', response)
    output_dfa = llm_dfa.extract_label_sequence(response)
    dfa_state_separated = llm_dfa.dfa_extract_dimensions(output_dfa)
    transition_indices_list = llm_dfa.transition_index(dfa_state_separated)
    hint_from_GPT = llm_dfa.hint_gen_from_dfa(dfa_state_separated, transition_indices_list)
    print('hint from GPT is:', hint_from_GPT)
    pysat_hints = hint_from_GPT[:2]
    # exit()

    # hints
    hint_dfas = None
    if pysat_hints is not None:
        
        # hint_dfas = [al_utils.gen_sup_hint_dfa(symbols) for symbols in pysat_hints]
        hint_dfas = list(itertools.chain.from_iterable(
            al_utils.gen_hints(symbols) for symbols in pysat_hints
        ))
    # exit()

    new_traces = Traces(set(), set())
    print('new traces:', new_traces)
    # exit()
    if isinstance(num_times, int):
        num_times = range(num_times)
    elif isinstance(num_times, tuple):
        num_times = range(*num_times)

    print('num times:', num_times)
    # exit()
    for t_i,t in enumerate(num_times):

        LIVETESTER = LiveTester(curriculum,
            show = (len(num_times)<=1),
            keep_open = True, # TODO remove
            # keep_open = show_plots,
            label = f"{run_name} - iteration {t} ({t_i+1}/{len(num_times)})",
            filebasename = run_name,
        ).start()
        task_timer = Timer()
        al_data = {
            "step": [],
            "pos": [],
            "neg": [],
            "time": [],
        }

        # Setting the random seed to 't'
        # exit()
        random.seed(t)
        open('./automata_learning_utils/data/data.txt','w').close
        open('./automata_learning_utils/data/automaton.txt','w').close
        # exit()


        # Reseting default values
        curriculum.restart()

        num_episodes = 0
        total = 0
        learned = 0
        step = 0
        enter_loop = 1
        num_conflicting_since_learn = 0
        update_rm = 0
        refreshed = 0
        testing_step = 0
        LIVETESTER.add_bool(step, 'learned', learned)

        # computing rm
        LIVETESTER.add_event(step, 'rm_update', force_update=show_plots)
        hm_file        = './automata_learning_utils/data/rm0.txt'
        hm_file_update = './automata_learning_utils/data/rm.txt'
        # exit()
        
        shutil.copy('./automata_learning/hypothesis_machine.txt', hm_file)
        # exit()
        
        # if hint_dfas is not None:
        #     print("Initial reward machine...")
        #     rm0 = al_utils.gen_dfa_from_hints(sup_hints=hint_dfas, show_plots=show_plots) # hint since BEGINNING
        #     # rm0 = al_utils.gen_partial_hint_dfa(pysat_hints[0], show_plots=show_plots) # begin with PARTIAL hint
        #     # rm0 = al_utils.gen_empty_hint_dfa(pysat_hints[0], show_plots=show_plots) # INITEMPTY
        #     rm0.export_as_reward_automaton(hm_file)
        #     Traces.rm_trace_to_symbol(hm_file)
        #     Traces.fix_rmfiles(hm_file)
        shutil.copy(hm_file, hm_file_update)
        print('curriculium:', curriculum)
        # exit()
        print('curric', curriculum.get_current_task())
        print('tester param', tester.get_task_params(curriculum.get_current_task()))
        # exit()
        
        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        # print(tester.params)
        print('task aux:', task_aux)
        print('curriculium get current task:', curriculum.get_current_task())
        # exit()
        num_features = len(task_aux.get_features())
        num_actions  = len(task_aux.get_actions())
        # q = np.zeros([1681,15,4])
        q = np.zeros([testing_params.agents_num,1681,15,num_actions])
        print('q dim is', q.shape)
        # exit()
        hypothesis_machine = tester.get_hypothesis_machine()
        tester_learned.update_hypothesis_machine_file(hm_file)
        tester_learned.update_hypothesis_machine()
        # LIVETESTER.add_event(step, 'rm_update') # already added

        # Task loop
        automata_history = []
        rewards = list()
        episodes = list()
        steps = list()
        testing_reward = 0 #initializes value
        all_traces = Traces(set(),set())
        LIVETESTER.add_traces_size(step, all_traces, 'all_traces')
        LIVETESTER.add_traces_size(step, new_traces, 'new_traces')
        epsilon = 0.3
        tt=t+1
        print("run index:", +tt)

        while not curriculum.stop_learning():
            num_episodes += 1

            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file_truth = '../experiments/craft/reward_machines/t1.txt'

            # Running 'task_rm_id' for one episode

            if learned==0:
                rm_file_learned = hm_file
                if update_rm:
                    update_rm = 0
                    refreshed = 1
                    tester_learned.update_hypothesis_machine_file(hm_file)
                    tester_learned.update_hypothesis_machine()
                    LIVETESTER.add_event(step, 'rm_refresh')
                    all_traces = Traces(set(),set())
                    LIVETESTER.add_traces_size(step, all_traces, 'all_traces')
                    num_conflicting_since_learn = 0
                    # q = np.zeros([1681,15,4])
                    q = np.zeros([testing_params.agents_num,1681,15,num_actions])

                    enter_loop = 1
            elif update_rm:
                rm_file_learned = hm_file_update

                task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
                num_features = len(task_aux.get_features())
                num_actions = len(task_aux.get_actions())
                rm_learned = tester_learned.get_hypothesis_machine() # used to be rm_learned = tester_learned.get_reward_machines()[0]
                if len(rm_learned.U)<16:
                    print("number of states:" + str(len(rm_learned.U)))
                else:
                    update_rm = 0
                    refreshed = 1
                    tester_learned.update_hypothesis_machine_file(hm_file)
                    tester_learned.update_hypothesis_machine()
                    LIVETESTER.add_event(step, 'rm_refresh')
                    all_traces = Traces(set(), set())
                    LIVETESTER.add_traces_size(step, all_traces, 'all_traces')
                    num_conflicting_since_learn = 0
                    # q = np.zeros([1681, 15, 4])
                    q = np.zeros([testing_params.agents_num,1681,15,num_actions])

                    enter_loop = 1
                    learned = 0
                    LIVETESTER.add_bool(step, 'learned', learned)

                update_rm = 0

            else:
                pass
            automata_history.append(rm_file_learned) #####fix this
            print('rm learned', rm_file_learned)

            epsilon = epsilon*0.99

            task_timer.resume()
            all_events, found_reward, stepcount, conflicting, \
                testing_reward, is_test, q = run_aqrm_task(
                epsilon, rm_file_truth, rm_file_learned, tester,
                  tester_learned, curriculum, show_print, learned, 
                  step, testing_reward, q, testing_params)
            
            task_timer.stop()
            LIVETESTER.add_bool(step, 'conflicting', conflicting)
            LIVETESTER.add_bool(step, 'is_positive', found_reward>0)
            LIVETESTER.add_bool(step, 'is_test', is_test)
            # print(",".join(all_events), "\n") #################################################


            #set up traces; we remove anything foreign to our ground truth formula

            if tester.game_type=="officeworld":
                while 'h' in all_events:
                   all_events.remove('h')
            elif tester.game_type=="trafficworld":
                while 'f' in all_events:
                   all_events.remove('f')
                while 'g' in all_events:
                   all_events.remove('g')
            elif tester.game_type=="craftworld":
                while 'd' in all_events:
                   all_events.remove('d')
                while 'g' in all_events:
                   all_events.remove('g')
                while 'h' in all_events:
                   all_events.remove('h')

            while '' in all_events:
                all_events.remove('')
                
            ## FOr loop for each agent ##
            if (conflicting==1 or refreshed==1):
                all_traces.add_trace(all_events, found_reward, learned)
                LIVETESTER.add_traces_size(step, all_traces, 'all_traces')

            if (num_episodes%100==0):
                print("run index:", +tt)
                toprint = "Total training reward at "+str(step)+": "+str(total)
                print(toprint)

            if num_episodes>5000:
                num_episodes

            total += found_reward
            step += stepcount
            num_conflicting_since_learn += conflicting
            rewards.append(found_reward)
            episodes.append(num_episodes)
            steps.append(step)

            if is_test:
                testing_step += testing_params.test_freq
                plot_dict.setdefault(testing_step, [])
                plot_dict[testing_step].append(testing_reward)
                LIVETESTER.add_reward(testing_step, testing_reward)


            if learned==1:

                if num_episodes%learning_params.relearn_period==0 and (num_conflicting_since_learn>0):
                    enter_loop = 1

                if conflicting==1:
                    new_traces.add_trace(all_events, found_reward, learned)
                    LIVETESTER.add_traces_size(step, new_traces, 'new_traces')
            print('all_events in run_aqrm_experiments:', all_events)
            print('new_traces in aqrm:', new_traces)
            # exit()

            # if enter_loop:
            #     print("\x1B[1;31;44m enter loop (%d positives) \x1B[m" % len(all_traces.positive))
            if (len(all_traces.positive)<learning_params.enter_loop) and enter_loop:
                LIVETESTER.add_event(step, 'rm_learn_failed', force_update=show_plots)
            if (len(all_traces.positive)>=learning_params.enter_loop) and enter_loop:
                LIVETESTER.add_event(step, 'rm_learn', force_update=show_plots)

                # positive = set()
                # negative = set()
                #
                # if learned==0:
                #     if len(all_traces.positive)>0:
                #         for i in list(all_traces.positive):
                #             if all_traces.symbol_to_trace(i) not in positive:
                #                 positive.add(all_traces.symbol_to_trace(i))
                #     if len(all_traces.negative)>0:
                #         for i in list(all_traces.negative):
                #             if all_traces.symbol_to_trace(i) not in negative:
                #                 negative.add(all_traces.symbol_to_trace(i))
                # else:
                #     if len(new_traces.positive)>0:
                #         for i in list(new_traces.positive):
                #             if new_traces.symbol_to_trace(i) not in positive:
                #                 positive.add(new_traces.symbol_to_trace(i))
                #     if len(new_traces.negative)>0 and len(all_traces.negative):
                #         for i in list(new_traces.negative):
                #             if new_traces.symbol_to_trace(i) not in negative:
                #                 negative.add(new_traces.symbol_to_trace(i))
                """equivalent:"""
                traces = all_traces if not learned else new_traces
                positive = set(Traces.symbol_to_trace(i) for i in traces.positive)
                negative = set(Traces.symbol_to_trace(i) for i in traces.negative)
                print('positive:', positive)
                print('negative:', negative)
                # exit()
                # print("PPP", positive) ####################################""
                # print("NNN", negative)


                positive_new = set() ## to get rid of redundant prefixes
                negative_new = set()

                if not learned:
                    for ptrace in positive:
                        new_trace = list()
                        previous_prefix = None #arbitrary
                        for prefix in ptrace:
                            if prefix != previous_prefix:
                                print('prefix in not learned:', prefix)
                                new_trace.append(prefix)
                                print('new_trace in not learned:', new_trace)
                            previous_prefix = prefix
                            print('previous prefix in not learned:', previous_prefix)
                            
                        positive_new.add(tuple(new_trace))
                        print('positive new in not learned:', positive_new)
                    # exit()
                    for ntrace in negative:
                        new_trace = list()
                        previous_prefix = None #arbitrary
                        for prefix in ntrace:
                            if prefix != previous_prefix:
                                new_trace.append(prefix)
                            previous_prefix = prefix
                        negative_new.add(tuple(new_trace))
                    if tester.game_type=="trafficworld":
                        if len(negative_new)<50:
                            negative_to_store = negative_new
                        else:
                            negative_to_store = set(random.sample(negative_new, 50))
                    else:
                        negative_to_store = negative_new
                    positive_to_store = positive_new
                    negative_new = negative_to_store
                    positive_new = positive_to_store

                    negative = set()
                    positive = set()

                else:
                    for ptrace in positive:
                        new_trace = list()
                        for prefix in ptrace:
                            new_trace.append(prefix)
                        positive_to_store.add(tuple(new_trace))
                        positive_new = positive_to_store
                        negative_new = negative_to_store

                    for ntrace in negative:
                        new_trace = list()
                        for prefix in ntrace:
                            new_trace.append(prefix)
                        negative_to_store.add(tuple(new_trace))
                        positive_new = positive_to_store
                        negative_new = negative_to_store

                traces_numerical = Traces(positive_new, negative_new)
                traces_file = './automata_learning_utils/data/data.txt'
                traces_numerical.export_traces(traces_file)
                LIVETESTER.add_traces_size(step, traces_numerical, 'traces_numerical')

                if learned == 1:
                    shutil.copy('./automata_learning_utils/data/rm.txt', '../experiments/use_past/t2.txt')

                print('hint dfa', hint_dfas)
                # exit()
                al_utils.al_timer.reset()

                ########################################################
                # from automata_learning_utils.pysat import sat_data_file
                # import automata_learning_utils.pysat.sat_data_file as pyhints
                # from automata_learning_utils.pysat.sat_data_file import extract_samples_from_traces

                # alphabet, positive_sample, negative_sample = extract_samples_from_traces(traces_numerical)
                # counterexamples = [
                #     trace
                #     for trace in positive_sample
                #     if any(trace not in hint for hint in hint_dfas)  # assuming those are sup hints
                # ]
                # if len(counterexamples) > 0:
                #     print('found counter example')


                #     instruct_for_gpt = f"The provided DFA sequence {hint_from_GPT}\
                #       is not correct. It must first go to intersection No.1 (symbol 'a')\
                #         then going to intersection No.2 (symbol 'b').Create the DFA step by step;\
                #         however, only write the DFA, no explanation is needed.\
                #         "

                #     messages.append({'role': 'user', 'content': instruct_for_gpt})

                #     prompt = messages
                #     params = llm_dfa.set_open_params()
                #     gpt_output = llm_dfa.get_completion(params, prompt)
                #     response = gpt_output.choices[0].message.content
                #     print('response in aqrm:', response)
                #     output_dfa = llm_dfa.extract_label_sequence(response)
                #     dfa_state_separated = llm_dfa.dfa_extract_dimensions(output_dfa)
                #     transition_indices_list = llm_dfa.transition_index(dfa_state_separated)
                #     hint_from_GPT = llm_dfa.hint_gen_from_dfa(dfa_state_separated, transition_indices_list)

                #     pysat_hints = hint_from_GPT
                #     print('updated hint is:', hint_from_GPT)
                #     hint_dfas = list(itertools.chain.from_iterable(
                #         al_utils.gen_hints(symbols) for symbols in pysat_hints
                #     ))
                    # exit()
                    # hint_dfas = [...]  # get new hint from ChatGPT, using the new counterexamples
                #############################################################
    

                automaton_visualization_filename = al_utils.learn_automaton(traces_file,
                    show_plots,
                    automaton_learning_algorithm=al_alg_name,
                    pysat_algorithm=sat_alg_name,
                    sup_hint_dfas=hint_dfas,
                    output_reward_machine_filename=hm_file_update,
                )
                al_data["step"].append(step)
                al_data["pos"].append(len(traces_numerical.positive))
                al_data["neg"].append(len(traces_numerical.negative))
                al_data["time"].append(al_utils.al_timer.elapsed())
                # if al_utils.al_timer.elapsed() > 10: #TODO REMOVE
                #     shutil.copy(traces_file, '../plotdata/data{:d}{:d}_{}_{:02d}{:02d}.txt'.format(
                #         2, int(task_id),
                #         algorithm_name[4:].upper(),
                #         t, len(al_data["time"])-1,
                #     ))

                # t2 is previous, t1 is new
                Traces.rm_trace_to_symbol(hm_file_update)
                Traces.fix_rmfiles(hm_file_update)

                if learned == 0:
                    shutil.copy('./automata_learning_utils/data/rm.txt',
                                             '../experiments/use_past/t2.txt')

                tester_learned.update_hypothesis_machine_file(hm_file_update) ## NOTE WHICH TESTER IS USED
                tester_learned.update_hypothesis_machine()
                # LIVETESTER.add_event(step, 'rm_learn') # already added


                print("learning")
                parent_path = os.path.abspath("../experiments/use_past/")
                os.makedirs(parent_path, exist_ok=True)

                shutil.copy(hm_file_update, '../experiments/use_past/t1.txt')
                if tester.game_type == 'officeworld':
                    current_and_previous_rms = '../experiments/office/tests/use_previous_experience.txt'
                elif tester.game_type == 'craftworld':
                    current_and_previous_rms = '../experiments/craft/tests/use_previous_experience.txt'
                elif tester.game_type == 'trafficworld':
                    current_and_previous_rms = '../experiments/traffic/tests/use_previous_experience.txt'
                elif tester.game_type == 'taxiworld':
                    current_and_previous_rms = '../experiments/taxi/tests/use_previous_experience.txt'
                else:
                    raise NotImplementedError(tester.game_type)


                tester_current = Tester(learning_params,testing_params,current_and_previous_rms)




                learned = 1
                LIVETESTER.add_bool(step, 'learned', learned)
                enter_loop = 0
                num_conflicting_since_learn = 0
                update_rm = 1

            if num_episodes%learning_params.relearn_period==0:
                new_traces = Traces(set(), set())
                LIVETESTER.add_traces_size(step, new_traces, 'new_traces')

            # if (learned==1 and num_episodes==1000):
            #
            #     tester_learned.update_hypothesis_machine()
            #     LIVETESTER.add_event(step, 'rm_update')
            #
            #
            #
            #     shutil.copy(hm_file_update, '../experiments/use_past/t2.txt')
            #     if tester.game_type == 'officeworld':
            #         current_and_previous_rms = '../experiments/office/tests/use_previous_experience.txt'
            #     elif tester.game_type == 'craftworld':
            #         current_and_previous_rms = '../experiments/craft/tests/use_previous_experience.txt'
            #     else:
            #         current_and_previous_rms = '../experiments/traffic/tests/use_previous_experience.txt'
            #
            #
            #     tester_current = Tester(learning_params,testing_params,current_and_previous_rms)
            #
            #
            #     q_old = np.copy(q)
            #     for ui in range(len(tester_current.reward_machines[0].get_states())):
            #         if not tester_current.reward_machines[0]._is_terminal(ui):
            #             is_transferred = 0
            #             for uj in range(len(tester_current.reward_machines[1].get_states())):
            #                 if not tester_current.reward_machines[1]._is_terminal(uj):
            #                     if tester_current.reward_machines[0].is_this_machine_equivalent(ui,tester_current.reward_machines[1],uj):
            #                         for s in range(len(q)):
            #                             if sum(q_old[s][uj])>0:
            #                                 q_old
            #                             q[s][ui] = np.copy(q_old[s][uj])
            #                         is_transferred = 1
            #                     else:
            #                         if not is_transferred:
            #                             for s in range(len(q)):
            #                                 q[s][ui] = 0
            #     # for ui in range(len(tester_current.reward_machines[0].get_states())):
            #     #     for s in range(len(q)):
            #     #         q[s][ui] = 0


        # Backing up the results
        print('Finished iteration ',t)

        reward_step = None # first step at which G(reward=1)
        for step,rwds in plot_dict.items():
            reward = rwds[-1]
            if not reward:
                reward_step = None
            elif reward_step is None:
                reward_step = step

        with open(details_filename, 'a') as f:
            wr = csv.writer(f)
            wr.writerow(["ITERATION_DETAIL:", t])
            wr.writerow(["task_time:", task_timer.elapsed()])
            wr.writerow(["al_step:", *al_data["step"]])
            wr.writerow(["al_pos:",  *al_data["pos"]])
            wr.writerow(["al_neg:",  *al_data["neg"]])
            wr.writerow(["al_time:", *al_data["time"]])
            wr.writerow(["total_time:", sum((task_timer.elapsed(),*al_data["time"]))])
            wr.writerow(["reward_step:", reward_step])

        LIVETESTER.close()

    # Showing results

    prc_25 = list()
    prc_50 = list()
    prc_75 = list()


    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()
    steps_plot = list()

    for step in plot_dict.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict[step]),25))
            current_50.append(np.percentile(np.array(plot_dict[step]),50))
            current_75.append(np.percentile(np.array(plot_dict[step]),75))
            current_step.append(sum(plot_dict[step])/len(plot_dict[step]))

        rewards_plot.append(sum(plot_dict[step])/len(plot_dict[step]))
        prc_25.append(sum(current_25)/len(current_25))
        prc_50.append(sum(current_50)/len(current_50))
        prc_75.append(sum(current_75)/len(current_75))
        steps_plot.append(step)



    # tester.plot_performance(steps_plot,prc_25,prc_50,prc_75) #TODO: uncomment
    # tester.plot_this(steps_plot,rewards_plot) #TODO: uncomment

    output_filename = f"../plotdata/{run_name}_{t}.csv"

    with open(output_filename, 'w') as f:
        wr = csv.writer(f)
        wr.writerows(list(plot_dict.values()))


    avg_filename = f"../plotdata/avgreward_{run_name}.txt"

    with open(avg_filename, 'w') as f:
        f.write("%s\n" % str(sum(rewards_plot) / len(rewards_plot)))
        for item in rewards_plot:
            f.write("%s\n" % item)
