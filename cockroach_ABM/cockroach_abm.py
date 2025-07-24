import numpy as np

class Cockroach_agent_basic:
    def __init__(self, nD: int, t_max: int):
        self.nD = nD # Number of distractors
        self.state = np.random.randint(self.nD+1) # Initial state
        self.states = np.zeros(t_max) # Past history of agent states
        self.states[0] = self.state
        self.dt = 0.1

    def update_state(self, t, shelter_numbers, shelter_sizes, theta, parameters):
        shelter_density = shelter_numbers[self.state]/shelter_sizes[self.state]
        leaving_rate = parameters['h']*theta[self.state]/(1+parameters['rho']*shelter_density**2)
        p_stay = np.exp(-leaving_rate*self.dt)
        if np.random.rand() > p_stay:  # If agent leaves
            new_state = np.random.randint(0, self.nD)  # Pick a new random shelter
            if new_state >= self.state:
                new_state += 1  # Ensure it's different from the current state
            self.state=new_state
        self.states[t] = self.state  # Store state

def tally_agents(agents):
    states = np.array([agent.state for agent in agents])
    return np.bincount(states, minlength=agents[0].nD + 1)

def update_agents(agents, t, past_shelter_numbers, shelter_sizes: np.array, theta: np.array, parameters: dict):
    # Update positions of all agents
    for agent in agents:
        agent.update_state(t, past_shelter_numbers, shelter_sizes, theta, parameters)

def simulate(agents, s, theta, parameters, t_max=1000):
    shelter_numbers_over_time = np.zeros((t_max, agents[0].nD+1))
    past_shelter_numbers = np.zeros(agents[0].nD+1)
    for t in range(t_max):
        shelter_numbers = tally_agents(agents)
        shelter_numbers_over_time[t] = shelter_numbers
        past_shelter_numbers[:] = shelter_numbers
        update_agents(agents, t, past_shelter_numbers, s, theta, parameters)
        
    return shelter_numbers_over_time
