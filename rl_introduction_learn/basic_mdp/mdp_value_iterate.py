#!/usr/bin/env python
# -*- coding:utf-8 -*-

#import gym
import random
#import numpy as np

class GriDMdp:
    def __init__(s):
        s.gamma = 0.9
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

def value_iterate(grid_mdp):
    for i in range(1000):
        print "---------------iter(%s)----------------" % str(i)
        delta = 0.0
        for state in grid_mdp.states:
            if state in grid_mdp.terminate_states:
                continue
            greedy_action = ""
            greedy_q_state_action = float('-Inf')
            for action in grid_mdp.trans[state]:
                next_state, state_reward, is_terminate, return_info = grid_mdp.transform(state, action)
                q_state_action = state_reward + grid_mdp.gamma * grid_mdp.v[next_state]
                if q_state_action > greedy_q_state_action:
                    greedy_action = action
                    greedy_q_state_action = q_state_action
            delta += abs(grid_mdp.v[state] - greedy_q_state_action)
            grid_mdp.v[state] = greedy_q_state_action
            grid_mdp.pi[state] = greedy_action
        grid_mdp.print_states()
        if delta < 1e-6:
            break
grid_mdp = GriDMdp()
value_iterate(grid_mdp)
