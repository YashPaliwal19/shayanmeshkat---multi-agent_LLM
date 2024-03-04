if __name__ == '__main__':
    # This is a terrible hack just to be able to execute this file directly
    import sys
    sys.path.insert(0, '../')

from worlds.game_objects import Actions, ActionsNew
import random, math, os
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class OfficeWorldParams:
    def __init__(self):
        pass

class OfficeWorld:

    def __init__(self, params):
        self._load_map()
        self.env_game_over = False

    def execute_action(self, a,l,c,ped):
        """
        We execute 'action' in the game
        """
        # x, y, l_, c_, p_ = self.agent
        x, y = self.agent[:2]
        temp = self.agent
        print('temp is', temp)
        temp_list = list(temp)
        print('temp list is', temp_list)
        temp_list[2:] = l, c, ped
        print('temp list value assiged is', temp_list)
        self.agent = tuple(temp_list)
        print('self agent after conversion to tuple is', self.agent)
        # executing action
        self.agent = self.xy_MDP_slip(a,0.9, l, c, ped) # progresses in x-y system

    def xy_MDP_slip(self,a,p, l ,c, ped):
        x,y,l, c, ped = self.agent
        slip_p = [p,(1-p)/2,(1-p)/2]
        check = random.random()

        # up    = 0
        # right = 1 
        # down  = 2 
        # left  = 3 

        if (check<=slip_p[0]):
            a_ = a

        elif (check>slip_p[0]) & (check<=(slip_p[0]+slip_p[1])):
            if a == 0: 
                a_ = 3
            elif a == 2: 
                a_ = 1
            elif a == 3: 
                a_ = 2
            elif a == 1: 
                a_ = 4
            elif a == 4: 
                a_ = 0

        else:
            if a == 0: 
                a_ = 1
            elif a == 2: 
                a_ = 3
            elif a == 3: 
                a_ = 4
            elif a == 4:
                a_ = 0
            elif a == 1: 
                a_ = 2

        action_ = ActionsNew(a_)
        if (x,y,action_) not in self.forbidden_transitions:
            if action_ == ActionsNew.stop:
                y = y
                x = x
            if action_ == ActionsNew.up:
                y+=1
            if action_ == ActionsNew.down:
                y-=1
            if action_ == ActionsNew.right:
                x+=1
            if action_ == ActionsNew.left:
                x-=1

        self.a_ = a_
        return (x,y, l, c , ped)

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        # print('self.agent is', self.agent)
        # exit()
        return self.actions

    def get_last_action(self):
        """
        Returns agent's last action
        """
        return self.a_

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret

    def get_state(self):
        return None # we are only using "simple reward machines" for the craft domain

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        print('self.agent is', self.agent)
        x,y,l,p,c = self.agent
        N,M, l_dim, p_dim, c_dim = 7, 6, 2, 2, 2
        ret = np.zeros((N,M,l_dim,p_dim,c_dim), dtype=np.float64)
        ret[x,y,l,p,c] = 1
        return ret.ravel() # from 2D to 1D (use a.flatten() is you want to copy the array)


    # def show(self):
    #     for y in range(8,-1,-1):
    #         if y % 3 == 2:
    #             for x in range(12):
    #                 if x % 3 == 0:
    #                     print("_",end="")
    #                     if 0 < x < 11:
    #                         print("_",end="")
    #                 if (x,y,ActionsNew.right) in self.forbidden_transitions:
    #                     print("_",end="")
    #                 else:
    #                     print(" ",end="")
    #             print()                
    #         for x in range(12):
    #             if (x,y,ActionsNew.left) in self.forbidden_transitions:
    #                 print("|",end="")
    #             elif x % 3 == 0:
    #                 print(" ",end="")
    #             if (x,y) == self.agent:
    #                 print("A",end="")
    #             elif (x,y) in self.objects:
    #                 print(self.objects[(x,y)],end="")
    #             else:
    #                 print(" ",end="")
    #             if (x,y,ActionsNew.right) in self.forbidden_transitions:
    #                 print("|",end="")
    #             elif x % 3 == 2:
    #                 print(" ",end="")
    #         print()      
    #         if y % 3 == 0:      
    #             for x in range(12):
    #                 if x % 3 == 0:
    #                     print("_",end="")
    #                     if 0 < x < 11:
    #                         print("_",end="")
    #                 if (x,y,ActionsNew.stop) in self.forbidden_transitions:
    #                     print("_",end="")
    #                 else:
    #                     print(" ",end="")
    #             print()    

    def show(self):
        rows = 6
        cols = 7

        x_coordinates = [(0, 0), (0, 1), (0, 3), (0, 5),
                          (3, 0), (3, 1), (3, 3), (3, 5), 
                          (5, 0), (5, 1), (5, 3), (5, 5),
                          (6, 0), (6, 1), (6, 3), (6, 5)]
        

        # Print the top border
        print("_" * (cols * 4 - 1))

        for i in range(rows):
            for j in range(cols):
                if (j, i) == self.agent[:2]:
                    print("| ^", end=" ")  # "^" at agent's position
                elif (j, i) in x_coordinates:
                    print("| X", end=" ")  # "X" at specified coordinates
                else:
                    print("|  ", end=" ")  # Empty space for other cells

            print("|")

            # Print the horizontal line between rows
            print("_" * (cols * 4 - 1))
        
        # for x in range(12):
        #         if (x,y,ActionsNew.left) in self.forbidden_transitions:
        #             print("|",end="")
        #         elif x % 3 == 0:
            

    # The following methods create the map ----------------------------------------------
    # def _load_map_old(self):
    #     # Creating the map
    #     self.objects = {}
    #     #env.agent = tuple([2, 2])
    #     #env.coffee = tuple([3, 5])
    #     #env.init_agent = tuple([2, 2])
    #     #env.locations = {(1, 1): 'a', (10, 1): 'b', (7, 3): 'c', (7, 4): 'e', (3, 5): 'f', (4, 4): 'g', (1, 8): 'd'}
    #     #env.mail = tuple([7, 4])
    #     self.objects[(1,1)] = "a"
    #     self.objects[(10,1)] = "b"
    #     #self.objects[(10,7)] = "c"
    #     self.objects[(1, 3)] = "c"
    #     #self.objects[(1,7)] = "d"
    #     self.objects[(7,4)] = "e"  # MAIL
    #     #self.objects[(8,2)] = "f"  # COFFEE
    #     self.objects[(3,5)] = "f"  # COFFEE
    #     self.objects[(4,4)] = "g"  # OFFICE

    #     # Adding walls
    #     self.forbidden_transitions = set()
    #     # for x in range(12):
    #     #     for y in [0]:
    #     #         self.forbidden_transitions.add((x,y,Actions.down))
    #     #     for y in [8]:
    #     #         self.forbidden_transitions.add((x,y,Actions.up))
    #     # for y in range(9):
    #     #     for x in [0]:
    #     #         self.forbidden_transitions.add((x,y,Actions.left))
    #     #     for x in [11]:
    #     #         self.forbidden_transitions.add((x,y,Actions.right))
    #     # general grid

    #     for x in [2, 5]:
    #         for y in [0, 1, 4, 5]: # No right turn to buildings
    #             self.forbidden_transitions.add((x,y,ActionsNew.right))

    #     for x in [1, 4]:
    #         for y in [0, 1, 4, 5]: # No left turn to buildings
    #             self.forbidden_transitions.add((x,y,ActionsNew.left))

    #     for y in [3]:
    #         for y in [0, 3, 6]: # No up turn to buildings
    #             self.forbidden_transitions.add((x,y,ActionsNew.up))

    #     for y in [2]:
    #         for y in [0, 3, 6]: # No down turn to buildings
    #             self.forbidden_transitions.add((x,y,ActionsNew.down))

    #     for x in range(7):
    #         for y in [0]: # wall, going down prohibited
    #             self.forbidden_transitions.add((x,y,ActionsNew.down))
        
    #     for x in range(7):
    #         for y in [5]: # wall, going up prohibited
    #             self.forbidden_transitions.add((x,y,ActionsNew.up))

    #     for x in [0]:
    #         for y in range(6): #going left prohibited
    #             self.forbidden_transitions.add((x,y,ActionsNew.left))

    #     for x in [6]:
    #         for y in range(6): #going right prohibited
    #             self.forbidden_transitions.add((x,y,ActionsNew.right))

    #     # Adding the agent
    #     self.agent = (2,1, 0, 0, 0)  
    #     self.actions = [ActionsNew.stop.value,ActionsNew.right.value,ActionsNew.stop.value,ActionsNew.left.value]
    
    def _load_map(self):
        # Creating the map
        self.objects = {}
        # self.objects[(1,1)] = "a"
        # self.objects[(10,1)] = "b"
        # #self.objects[(10,7)] = "c"
        # self.objects[(1, 3)] = "c"
        # self.objects[(1,7)] = "d"
        # self.objects[(7,4)] = "e"  # MAIL
        # self.objects[(3,5)] = "f"  # COFFEE
        # self.objects[(4,4)] = "g"  # OFFICE

        self.objects[(4,3,1,0,0)] = "a"
        self.objects[(5,3,1,0,0)] = "b"
        self.objects[(6,3,0,0,0)] = "c"
        # self.objects[(2,2,1,0,0)] = "l"
        # # self.objects[(3,2,1,0,0)] = "!l|c"
        # self.objects[(4,2,0,0,0)] = "l"

        # Adding the agent
        self.agent = (2,0,0,0,0)
        self.actions = [ActionsNew.stop.value,ActionsNew.up.value,
                        ActionsNew.down.value,ActionsNew.right.value,
                        ActionsNew.left.value]

        # Adding walls
        self.forbidden_transitions = set()
        for x in [2, 4]:
            for y in [0, 1, 3, 5]: # No right turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.right))

        for x in [1, 4]:
            for y in [0, 1, 3, 5]: # No left turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.left))

        for y in [2, 4]:
            for x in [0, 1, 3, 5, 6]: # No up turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.up))

        for y in [2, 4]:
            for x in [0, 1, 3, 5, 6]: # No down turn to buildings
                self.forbidden_transitions.add((x,y,ActionsNew.down))

        for x in range(7):
            for y in [0]: # wall, going down prohibited
                self.forbidden_transitions.add((x,y,ActionsNew.down))
        
        for x in range(7):
            for y in [5]: # wall, going up prohibited
                self.forbidden_transitions.add((x,y,ActionsNew.up))

        for x in [0]:
            for y in range(6): #going left prohibited
                self.forbidden_transitions.add((x,y,ActionsNew.left))

        for x in [6]:
            for y in range(6): #going right prohibited
                self.forbidden_transitions.add((x,y,ActionsNew.right))
        
def play():
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w":ActionsNew.up.value,"d":ActionsNew.right.value,
                     "s":ActionsNew.down.value,"a":ActionsNew.left.value,
                     "q":ActionsNew.stop.value}
    params = OfficeWorldParams()

    # play the game!
    tasks = ["../../experiments/office/reward_machines/t%d.txt"%i for i in [1,3,4]]
    reward_machines = []
    for t in tasks:
        reward_machines.append(RewardMachine(t))
    for i in range(len(tasks)):
        print("Running", tasks[i])

        game = OfficeWorld(params) # setting the environment
        rm = reward_machines[i]  # setting the reward machine
        s1 = game.get_state()
        u1 = rm.get_initial_state()
        while True:
            # Showing game
            game.show()
            print("Events:", game.get_true_propositions())
            #print(game.getLTLGoal())
            # Getting action
            print("u:", u1)
            print("\nAction? ", end="")
            a = input()
            print()
            # Executing action
            if a in str_to_action:
                game.execute_action(str_to_action[a],0,0,0)

                # Getting new state and truth valuation
                s2 = game.get_state()
                events = game.get_true_propositions()
                u2 = rm.get_next_state(u1, events)
                r = rm.get_reward(u1,u2,s1,a,s2)
                
                # Getting rewards and next states for each reward machine
                rewards, next_states = [],[]
                for j in range(len(reward_machines)):
                    j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
                    rewards.append(j_rewards)
                    next_states.append(j_next_states)
                
                print("---------------------")
                print("Rewards:", rewards)
                print("Next States:", next_states)
                print("Reward:", r)
                print("---------------------")
                
                if game.env_game_over or rm.is_terminal_state(u2): # Game Over
                    break 
                
                s1 = s2
                u1 = u2
            else:
                print("Forbidden action")
        game.show()
        print("Events:", game.get_true_propositions())
    
# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
    play()
