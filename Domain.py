import random
import numpy as np
from ParticleBelief import ParticleBelief

# THIS FILE PROVIDES SKELETON CODE FOR DEFINING THE PLANNING PROBLEM
# DEPENDING ON THE DOMAIN CONSIDERED, SEVERAL FUNCTIONS IN THE MODEL CLASS NEED TO BE ADJUSTED/IMPLEMENTED

class ModelState:
    def __init__(self, time, location):

        # fully observable
        self.time = time

        # partially observable
        self.location = location

        # precompute state hash
        self.hash = hash(tuple([self.time, self.location]))

    def get_state_hash(self):
        return self.hash

    def __str__(self):
        return '<State(time={}, location={})>'.format(
            self.time, self.location)


class ModelObservation:
    def __init__(self, time, location, o):
        self.time = time
        self.location = location
        self.o = o
        # last action is not here, because for a given history in the tree they are fixed
        
        self.hash = hash(tuple([self.time, self.location, self.o]))

    def get_observation_hash(self):
        return self.hash

    def __str__(self):
        return '<Observation(time={}, location={}, o={})>'.format(
            self.time, self.location, self.o)
        

class Model:
    def __init__(self, n=1000):
        self.n = n
        self.num_actions = 2
        
        self.free_energy_cache = {}
        self.belief_transition_cache = {}
        self.particle_filter_cache = {}
        self.expected_observations_cache = {}
        self.feasible_action_cache = {}
        self.map_observations = {}  # key: observation hash, value: observation object
        
    def reset_caches(self):
        self.free_energy_cache = {}
        self.belief_transition_cache = {}
        self.particle_filter_cache = {}
        self.expected_observations_cache = {}
        self.map_observations = {}
        self.feasible_action_cache = {}
        
    def get_map_action_names(self):
        return {0:'left', 1:'right'}
    
    def get_prior(self):
        # define initial state variables (fully observable)
        time = 0
        
        belief_prior = ParticleBelief()
        for i in range(500): 
            state = ModelState(time, 0)
            belief_prior.states.append(state)
        for i in range(500):
            state = ModelState(time, 1)
            belief_prior.states.append(state)
            
        return belief_prior
        
    def get_feasible_actions(self, b):
        assert isinstance(b, ParticleBelief)        
        timestep = b.states[0].time
        actions = []
        
        # TODO determine which actions are feasible in b and add them to action list
        
        return actions
        
    
    def get_state_transition(self, s, a):
        assert isinstance(s, ModelState)
        assert a>=0 and a<self.num_actions
        
        # 0 = left
        # 1 = right
        
        next_state = None
        
        #if a == 0:
            # ext_state = ...
            
        # TODO implement transition function
            
        assert not (next_state is None)
        return next_state
    
    def _get_obs_probs(self, s):
        probs = []
        # TODO determine probability for each observation given s
        return probs
    
    def get_observation_probability(self, s, o):
        assert isinstance(s, ModelState)
        assert isinstance(o, ModelObservation)
        
        # TODO compute probability to observe o in s and return it
        
        return 0.0
    
    def get_observation(self, s):
        assert isinstance(s, ModelState)        
        o = -1
                
        # TODO sample an observation given s
            
        return ModelObservation(s.time, s.spot_location, s.rock_location, o)
    
    def get_observation_prior(self, o):
        assert isinstance(o, ModelObservation)
        # return prior preference associated with o
        return 0.0
    
    def get_expected_observations(self, b, n=1000):
        # this function can be used to estimate the expectation over observations for belief b, it returns a set containing n observations
        assert isinstance(b, ParticleBelief)

        cache_key = hash(tuple(b.history))
        if len(b.history) > 0 and cache_key in self.expected_observations_cache:
            return self.expected_observations_cache[cache_key]

        obs = []
        for i in range(n):
            state_idx = np.random.randint(len(b.states))
            s = b.states[state_idx]
            o = self.get_observation(s)
            obs.append(o)

        self.expected_observations_cache[cache_key] = obs
        return obs
    
    def get_reward(self, b, n=1000):
        reward = -1.0 * self.get_expected_free_energy(b, n)
        return reward
    
    def get_expected_free_energy(self, b, n=1000):
        assert isinstance(b, ParticleBelief)
        # this function estimates the instantaneous expected free energy in belief b
        # it computes the same as get_instant_expected_free_energy, but it's based on a particle belief representation

        cache_key = hash(tuple(b.history))
        if len(b.history) > 0 and cache_key in self.free_energy_cache:
            return self.free_energy_cache[cache_key]

        # first term
        first_term = 0.0
        observations = self.get_expected_observations(b, n)
        o_hash_keys = [o.get_observation_hash() for o in observations]
        o_hash_keys_unique = np.unique(o_hash_keys)

        map_observations = {}
        for o in observations:
            map_observations[o.get_observation_hash()] = o

        for k in o_hash_keys_unique:
            observation = map_observations[k]
            observation_probability = o_hash_keys.count(k) / len(o_hash_keys)
            observation_prior = self.get_observation_prior(observation)
            #print(observation_probability, observation_prior, observation_probability * (np.log(observation_probability + 1e-3) - np.log(observation_prior + 1e-3)))
            first_term += observation_probability * (np.log(observation_probability + 1e-3) - np.log(observation_prior + 1e-3))
        first_term = max(first_term , 0.0)

        # second term
        second_term = 0.0
        b_hash_keys = [s.get_state_hash() for s in b.states]
        b_hash_keys_unique = np.unique(b_hash_keys)

        map_states = {}
        for s in b.states:
            map_states[s.get_state_hash()] = s

        for k in b_hash_keys_unique:
            state = map_states[k]
            state_belief = b_hash_keys.count(k) / len(b_hash_keys)

            # sample observations for this state
            observations = []
            for i in range(n):
                o = self.get_observation(state)
                observations.append(o)

            # estimate \sum_o P(o|state) log[P(o|state)]
            prob_log_sum = 0.0
            o_hash_keys = [o.get_observation_hash() for o in observations]
            o_hash_keys_unique = np.unique(o_hash_keys)
            for k in o_hash_keys_unique:
                prob_o = o_hash_keys.count(k) / len(o_hash_keys)
                prob_log_sum += max(prob_o * np.log(prob_o + 1e-3) * -1.0, 0.0)

            # multiply by state belief and add to G
            second_term += state_belief * prob_log_sum
        second_term = max(second_term, 0.0)
            
        # free energy for world state
        third_term = 0.0
        world_state_variables = ['rock_state'] # include state variables for which information needs to be gathered
        for var in world_state_variables:
            values = [getattr(s, var) for s in b.states]            
            value_counts = [(i, values.count(i)) for i in set(values)]
            min_fe = np.inf
            for v in value_counts:
                val_count = v[1]
                prob_val = val_count / len(values)
                fe = (-1 * prob_val * np.log(self.pref_correct)) + (-1 * (1.0 - prob_val) * np.log((1.0-self.pref_correct) + 1e-3))
                if fe < min_fe:
                    min_fe = fe
            free_energy = min_fe
            third_term += free_energy
            
        G = first_term + second_term + third_term
        self.free_energy_cache[cache_key] = G

        return G

    def random_choice(self, options, probs):
        # this function samples an option based on the associated probabilities
        x = np.random.rand()
        cum = 0
        for i, p in enumerate(probs):
            cum += p
            if x < cum:
                break
        return options[i]
    
    def execute_weighted_particle_filter_transition(self, b, a, n=1000):
        assert isinstance(b, ParticleBelief)
        # pseudocode in MONTE CARLO SAMPLING METHODS FOR APPROXIMATING I-POMDPS
        
        # sample an observation o by executing a in b
        belief_size = len(b.states)
        state_idx = int(belief_size * random.random())
        s = b.states[state_idx]
        s_next = self.get_state_transition(s, a)
        o = self.get_observation(s_next)
        assert o != None
        
        # check the cache
        observation_hash = o.get_observation_hash()
        cache_key = hash(tuple(b.history + [a, observation_hash]))
        if len(b.history) > 0 and cache_key in self.particle_filter_cache:
            return self.particle_filter_cache[cache_key], o
        
        # apply weighted particle filter to approximate b_a^o
        successor_states = []
        sample_weights = []
        weight_sum = 0.0
        for s in b.states:
            s_next = self.get_state_transition(s, a)
            weight = self.get_observation_probability(s_next, o)
            weight_sum += weight
            successor_states.append(s_next)
            sample_weights.append(weight)
        assert weight_sum > 0.0, 'observation was not seen while sampling observations from successor states'
        
        # sample with replacement based on the normalized weights        
        belief = ParticleBelief()
        belief.states = random.choices(successor_states, weights=(np.array(sample_weights) / weight_sum), k=n)
        
        # set history in new belief
        history = b.history.copy()
        history.append(a)
        history.append(o.get_observation_hash())
        belief.history = history
        self.particle_filter_cache[cache_key] = belief
        
        return belief, o
    
    
    def is_termination_belief(self, b):
        assert isinstance(b, ParticleBelief)
        return False
    