#!/usr/bin/env python
# -*- coding:utf-8 -*-

#import gym
import random
import numpy as np

class GriDMdp:
    def __init__(s):
        s.gamma = 0.9
        s.alpha = 0.3
        s.epsilon = 0.1
        s.states = range(1,26) #状态空间
        s.actions = ['n', 'e', 's', 'w'] #动作空间
        s.terminate_states = {15:1.0, 4:-1.0, 9:-1.0, \
            11:-1.0, 12:-1.0, 23:-1.0, 24:-1.0, 25:-1.0} #结束状态
        s.trans = {} #状态下的动作空间
        for state in s.states:
            if not state in s.terminate_states:
                s.trans[state] = {}
        s.trans[1]['e'] = 2
        s.trans[1]['s'] = 6
        s.trans[2]['e'] = 3 
        s.trans[2]['w'] = 1
        s.trans[2]['s'] = 7
        s.trans[3]['e'] = 4
        s.trans[3]['w'] = 2
        s.trans[3]['s'] = 8
        s.trans[5]['w'] = 4
        s.trans[5]['s'] = 10
        s.trans[6]['e'] = 7
        s.trans[6]['s'] = 11
        s.trans[6]['n'] = 1
        s.trans[7]['e'] = 8
        s.trans[7]['w'] = 6 
        s.trans[7]['s'] = 12
        s.trans[7]['n'] = 2
        s.trans[8]['e'] = 9
        s.trans[8]['w'] = 7 
        s.trans[8]['s'] = 13
        s.trans[8]['n'] = 3
        s.trans[10]['w'] = 9
        s.trans[10]['s'] = 15
        s.trans[13]['e'] = 14
        s.trans[13]['w'] = 12 
        s.trans[13]['s'] = 18
        s.trans[13]['n'] = 8
        s.trans[14]['e'] = 15
        s.trans[14]['w'] = 13
        s.trans[14]['s'] = 19
        s.trans[14]['n'] = 9
        s.trans[16]['e'] = 17
        s.trans[16]['s'] = 21
        s.trans[16]['n'] = 11
        s.trans[17]['e'] = 18
        s.trans[17]['w'] = 16 
        s.trans[17]['s'] = 22
        s.trans[17]['n'] = 12
        s.trans[18]['e'] = 19
        s.trans[18]['w'] = 17 
        s.trans[18]['s'] = 23
        s.trans[18]['n'] = 13
        s.trans[19]['e'] = 20
        s.trans[19]['w'] = 18 
        s.trans[19]['s'] = 24
        s.trans[19]['n'] = 14
        s.trans[20]['w'] = 19
        s.trans[20]['s'] = 25
        s.trans[20]['n'] = 15
        s.trans[21]['e'] = 22
        s.trans[21]['n'] = 16
        s.trans[22]['e'] = 23
        s.trans[22]['w'] = 21
        s.trans[22]['n'] = 17
        
        s.rewards = {} #奖励
        for state in s.states:
            s.rewards[state] = {}
            for action in s.actions:
                s.rewards[state][action] = 0
                if state in s.trans and action in s.trans[state]:
                    next_state = s.trans[state][action]
                    if next_state in s.terminate_states:
                        s.rewards[state][action] = s.terminate_states[next_state]
        s.pi = {} #策略
        for state in s.trans:
            s.pi[state] = random.choice(s.trans[state].keys())
        s.last_pi = s.pi.copy()

        s.v = {} #状态值函数
        for state in s.states:
            s.v[state] = 0.0
    def get_random_action(s, state):
        s.pi[state] = random.choice(s.trans[state].keys())
        return s.pi[state]

    def transform(s, state, action):
        next_state = state
        state_reward = 0
        is_terminate = True
        return_info = {}

        if state in s.terminate_states:
            return next_state, state_reward, is_terminate, return_info
        if state in s.trans:    
            if action in s.trans[state]:
                next_state = s.trans[state][action]
        if state in s.rewards:
            if action in s.rewards[state]:
                state_reward = s.rewards[state][action]
        if not next_state in s.terminate_states:
            is_terminate = False
        return next_state, state_reward, is_terminate, return_info
    
    def print_states(s):
        for state in s.states:
            if state in s.terminate_states:
                print "*",
            else:
                print round(s.v[state], 2),
            if state % 5 == 0:
                print "|"

def epsilon_greey(state_action_value_dic, state, epsilon):
    if not state in state_action_value_dic:
        return "NULL", 0.0
    action_list = state_action_value_dic[state].keys()
    len_action = len(action_list) 
    action_prob = [epsilon / float(len_action)] * len_action
    max_val = float('-inf') 
    max_idx = -1
    for idx in range(len_action):
        action = action_list[idx]
        state_action_value = state_action_value_dic[state][action]
        if state_action_value > max_val:
            max_val = state_action_value
            max_idx = idx
    if max_idx < 0:
        return np.random.choice(action_list),action_prob[0]
    else:
        action_prob[max_idx] += (1 - epsilon)
        epsilon_greey_action = np.random.choice(action_list, p=action_prob)
        return epsilon_greey_action, action_prob[max_idx]
        
def td_sarsa_method(grid_mdp):
    "action-strategy is epsilon_greey strategy, target-strategy is epsilon_greey strategy"
    state_action_value_dic = {}
    for iter_idx in range(50000):
        #print "-----------------------"
        state = random.choice(grid_mdp.states)
        while(state in grid_mdp.terminate_states):
            state = random.choice(grid_mdp.states)
        sample_end = False
        while sample_end != True:
            if not state in state_action_value_dic:
                state_action_value_dic[state] = {}
            # action-strategy choose epsilon_greey strategy
            for action in grid_mdp.trans[state]:
                if not action in state_action_value_dic[state]:
                    state_action_value_dic[state][action] = 0.0    
            action, prob = epsilon_greey(state_action_value_dic, state, grid_mdp.epsilon)
            next_state, state_reward, is_terminate, return_info = grid_mdp.transform(state, action)
            # target-strategy choose epsilon_greey strategy
            next_state_action_value = 0.0
            next_action, next_prob = epsilon_greey(state_action_value_dic, next_state, grid_mdp.epsilon)
            if next_state in state_action_value_dic:
                if next_action in state_action_value_dic[next_state]:
                    next_state_action_value = state_action_value_dic[next_state][next_action]
            difference = state_reward +  grid_mdp.gamma * next_state_action_value - state_action_value_dic[state][action]
            state_action_value_dic[state][action] += grid_mdp.alpha * difference
            state = next_state
            sample_end = is_terminate

        if iter_idx % 1000 == 0:
            print "-"*18
            for state in sorted(state_action_value_dic.keys()):
                for action in sorted(state_action_value_dic[state]):
                        print state,action,state_action_value_dic[state][action]

grid_mdp = GriDMdp()
td_sarsa_method(grid_mdp)
