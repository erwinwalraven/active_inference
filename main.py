import numpy as np
import random
from ParticleBelief import ParticleBelief
from Domain import Model, ModelState, RewardType

# THIS FILE PROVIDES SKELETON CODE FOR PLANNING USING MCTS
        
class TreeNodeObservationSplit:
    def __init__(self, pomdp):
        assert isinstance(pomdp, Model)
        self.pomdp = pomdp
        self.childs = {}   # key is observation hash
        self.V = 0.0
        self.N = 0
        
    def ensure_child_exists(self, observation_hash, belief):
        assert isinstance(belief, ParticleBelief)
        assert isinstance(observation_hash, int)
        if not observation_hash in self.childs:
            child = TreeNode(self.pomdp, belief)
            self.childs[observation_hash] = child
            
    def get_num_nodes(self):
        num_nodes = 0
        for o in self.childs:
            num_nodes += self.childs[o].get_num_nodes()
        return 1 + num_nodes

class TreeNode:
    def __init__(self, pomdp, belief):
        assert isinstance(pomdp, Model)
        assert isinstance(belief, ParticleBelief)
        self.pomdp = pomdp
        self.belief = belief
        self.childs = {}   # key is observation hash
        self.N = 0
        self.reward = None
        
    def is_fully_expanded(self):
        return len(self.childs) > 0
    
    def create_childs(self):
        for a in range(self.pomdp.num_actions):
            self._add_child(a)
    
    def _add_child(self, a):
        assert a < self.pomdp.num_actions
        child = TreeNodeObservationSplit(self.pomdp)
        self.childs[a] = child
    
    def get_reward(self, n):
        if self.reward == None:
            self.reward = self.pomdp.get_reward(self.belief, n)
        return self.reward
    
    def get_num_nodes(self):
        num_nodes = 0
        for a in self.childs:
            num_nodes += self.childs[a].get_num_nodes()
        return 1 + num_nodes
    
    
def rollout(pomdp, belief, depth, max_depth, reward_temperature, n):
    assert isinstance(belief, ParticleBelief)
    if depth == max_depth:
        return 0.0
    feasible_actions = pomdp.get_feasible_actions(belief)
    a = random.choice(feasible_actions)
    next_belief, o = pomdp.execute_weighted_particle_filter_transition(belief, a, n)    
    expected_free_energy = pomdp.get_reward(belief, n) * reward_temperature
    future_expected_free_energy = rollout(pomdp, next_belief, depth+1, max_depth, reward_temperature, n)
    return expected_free_energy + future_expected_free_energy

def random_choice(options, probs):
    x = np.random.rand()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]

def get_best_action_ucb(node, C):
    assert isinstance(node, TreeNode)
    
    feasible_actions = node.pomdp.get_feasible_actions(node.belief)
    
    # UCB
    best_action_id = -1
    best_action_value = -np.inf
    for a in feasible_actions:
        action_node = node.childs[a]
        child_value = 0.0
        if node.N == 0 or action_node.N == 0:
            child_value = np.inf
        else:
            child_value = node.childs[a].V + C * np.sqrt((2.0 * np.log(node.N)) / (node.childs[a].N))
        if child_value >= best_action_value:
            best_action_id = a
            best_action_value = child_value
    return best_action_id

def get_best_action_boltzmann(node, C):
    assert isinstance(node, TreeNode)
    
    # Boltzmann
    child_values = []
    for a in range(node.pomdp.num_actions):
        child_value = 0.0
        if not(node.N == 0 or node.childs[a].N == 0):
            child_value = node.childs[a].V + C * np.sqrt((2.0 * np.log(node.N)) / (node.childs[a].N))
        child_values.append(child_value)
    child_values = np.array(child_values)
    
    # normalize child values to [0,1] before applying exp
    child_values = child_values - np.min(child_values)
    child_value_diff = np.max(child_values) - np.min(child_values)
    if np.any(child_values) and child_value_diff > 0.0:
        child_values = child_values / child_value_diff
        
    # compute action values
    action_values = np.exp(child_values)
    
    # disable actions that are not feasible by setting their probability to 0
    feasible_actions = node.pomdp.get_feasible_actions(node.belief)
    for a in range(node.pomdp.num_actions):
        if a not in feasible_actions:
            feasible_actions[a] = 0.0
    
    action_probabilities = np.array(action_values) / np.sum(action_values)
    
    assert abs(np.sum(action_probabilities) - 1.0) < 1e-5, '{}'.format(action_values)
    best_action_id = random_choice([a for a in range(node.pomdp.num_actions)], action_probabilities)
    
    return best_action_id

def get_best_action(node, C):
    assert isinstance(node, TreeNode)
    return get_best_action_ucb(node, C)


def get_best_action_final(node):
    assert isinstance(node, TreeNode)
    feasible_actions = node.pomdp.get_feasible_actions(node.belief)
    best_action_id = -1
    best_action_value = -np.inf
    for a in feasible_actions:
        child_value = node.childs[a].V
        if child_value >= best_action_value:
            best_action_id = a
            best_action_value = child_value
    return best_action_id
    

def simulate(node, depth, max_depth, C, reward_temperature, n):
    assert isinstance(node, TreeNode)
    
    # return 0 if max depth was reached
    if depth == max_depth:
        return 0.0
    
    # if node does not have childs for the actions, we create them first and we do a rollout
    if not node.is_fully_expanded():
        node.create_childs()
        return rollout(node.pomdp, node.belief, depth, max_depth, reward_temperature, n)
    
    # get the belief b that corresponds to this node
    b = node.belief
    
    # select action and corresponding child
    a = get_best_action(node, C)
    
    # sample b' and o
    b_next, o = node.pomdp.execute_weighted_particle_filter_transition(b, a, n)
    observation_hash = o.get_observation_hash()
    
    # ensure that child exists for this observation
    best_action_child = node.childs[a]
    assert isinstance(best_action_child, TreeNodeObservationSplit)
    best_action_child.ensure_child_exists(observation_hash, b_next)

    # get child that corresponds to the belief that we encountered
    belief_child = best_action_child.childs[observation_hash]
    
    # determine return value
    ret_value = reward_temperature * node.get_reward(n) + simulate(belief_child, depth+1, max_depth, C, reward_temperature, n)
    
    # sanity check
    if ret_value < -500.0:
        print('very low value encountered in simulate at depth {}: {}'.format(depth, ret_value))
    
    # update values and visit counts
    node.N += 1
    best_action_child.N += 1
    best_action_child.V += ((ret_value - best_action_child.V) / best_action_child.N)
    
    return ret_value


def plan(node, depth, max_depth, num_simulations, C, reward_temperature, n):
    assert abs(reward_temperature-1.0) < 1e-5
    assert isinstance(node, TreeNode)
    for i in range(num_simulations):
        simulate(node, depth, max_depth, C, reward_temperature, n)
    return get_best_action_final(node)
        

if __name__ == "__main__":
    import datetime
    
    # config parameters
    num_simulations = 5000                 # this is the number of simulations for planning
    n = 500                                # this is the number of sampling runs for belief estimation
    C = 10.0                               # scalar for exploration bonus
    reward_temperature = 1.0               # scalar for expected free energy, should be 1.0
    num_experiment_runs = 100
    perform_random_actions = False
    
    # initialize your Model object here
    model = Model()
    map_action_names = model.get_map_action_names()
    num_steps = 10
    


    print('===== INITIAL BELIEF FOR PLANNING =====')
    belief_prior = model.get_prior()
    model.print_belief(belief_prior)
    
    # define true state here
    actual_state = ModelState()
    
    print()
    print('Actual state', actual_state)
    
    start_time = datetime.datetime.now()
    for i in range(num_steps):
        # check if belief is termination belief
        if model.is_termination_belief(belief_prior):
            break
        
        # run the planner
        tree_root = TreeNode(model, belief_prior)
        max_depth = num_steps + 1     # at this depth value=0 is used in the search tree
        a = plan(tree_root, i, max_depth, num_simulations, C, reward_temperature, n)
        execute_a = a
        
        if perform_random_actions:
            feasible_actions = model.get_feasible_actions(belief_prior)
            execute_a = feasible_actions[np.random.randint(0,len(feasible_actions))]
        
        print()
        print('===== PLAN ACTION =====')
        print('Depth:', i)
        print('Selected action:', execute_a, map_action_names[execute_a])
        print('Action values:')
        for action_id in tree_root.childs:
            print('', action_id, map_action_names[action_id], tree_root.childs[action_id].V)
        print('Action visit counts:', [tree_root.childs[action_id].N for action_id in tree_root.childs])
        print('Num nodes in tree:', tree_root.get_num_nodes())
        
        # execute the action in actual_state
        next_state = model.get_state_transition(actual_state, execute_a)
        
        # sample an observation given the successor state
        o = model.get_observation(next_state)
        
        print()
        print('===== EXECUTE IT =====')
        print('Observation:', o)
        node_next = tree_root.childs[execute_a].childs[o.get_observation_hash()]
        model.print_belief(node_next.belief)
        print()
        print('Next state', next_state)
        
        # prepare for next iteration
        belief_prior = node_next.belief
        actual_state = next_state
        