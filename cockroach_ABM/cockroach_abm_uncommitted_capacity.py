import numpy as np

class Cockroach_agent_uncommitted_capacity:
    def __init__(self, nD: int, t_max: int, uncommitted_timescale = 500):
        self.nD = nD # Number of distractors
        self.state = self.nD+1
         # Initial state
        self.states = np.zeros(t_max) # Past history of agent states
        self.states[0] = self.state
        self.dt = 0.1
        uncommitted_probability = np.exp(-self.dt/uncommitted_timescale)
        self.uncommitted_probabilities = np.ones(self.nD+2)*(1-uncommitted_probability)/(self.nD+1)
        self.uncommitted_probabilities[-1] = uncommitted_probability
        self.rejection_scaling = 1 # Scaling factor for whether shelter entry gets rejected

    def update_state(self, t, shelter_numbers, shelter_sizes, theta, parameters):
        
        # If agent is uncommitted
        if self.state == self.nD+1:

            # Try shelter randomly
            new_state = np.random.choice(np.arange(0, self.nD+2), p = self.uncommitted_probabilities)

            # If shelter is too packed, stay uncommitted
            if new_state != self.nD+1 and np.random.rand() > 1-self.rejection_scaling*shelter_numbers[new_state]/shelter_sizes[new_state]:
                new_state = self.nD+1

        # If agent is under shelter
        else: 
            new_state = self.state
            shelter_density = shelter_numbers[self.state]/shelter_sizes[self.state]
            leaving_rate = parameters['h']*theta[self.state]/(1+parameters['rho']*shelter_density**2)
            p_stay = np.exp(-leaving_rate*self.dt)
            if np.random.rand() > p_stay:  # If agent leaves
                new_state = self.nD+1
        self.state=new_state
        self.states[t] = self.state  # Store state

def tally_agents(agents):
    states = np.array([agent.state for agent in agents])
    return np.bincount(states, minlength=agents[0].nD + 2)

def update_agents(agents, t, past_shelter_numbers, shelter_sizes: np.array, theta: np.array, parameters: dict):
    # Update positions of all agents
    for agent in agents:
        agent.update_state(t, past_shelter_numbers, shelter_sizes, theta, parameters)

def simulate(agents, s, theta, parameters, t_max=1000):
    shelter_numbers_over_time = np.zeros((t_max, agents[0].nD+2))
    past_shelter_numbers = np.zeros(agents[0].nD+2)
    shelter_numbers_over_time[0] = tally_agents(agents)
    for t in range(1, t_max):
        shelter_numbers = tally_agents(agents)
        # print(shelter_numbers)
        shelter_numbers_over_time[t] = shelter_numbers
        past_shelter_numbers[:] = shelter_numbers
        update_agents(agents, t, past_shelter_numbers, s, theta, parameters)
        
    return shelter_numbers_over_time
