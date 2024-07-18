import numpy as np

class RobotHMM:
    def __init__(self, grid_size, transition_prob, observation_prob):
        # init: The constructor method initializes the grid size,
        # transition probabilities, observation probabilities, and the initial belief state.
        self.grid_size = grid_size
        self.transition_prob = transition_prob
        self.observation_prob = observation_prob
        self.belief_state = np.ones((grid_size, grid_size)) / (grid_size * grid_size)

    def predict(self):
        # predict: This method predicts the next belief state based on the transition model
        new_belief = np.zeros_like(self.belief_state)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        # For each valid new position (ni, nj), the new belief state
                        # is updated using the transition probabilities and the current belief state.
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            new_belief[ni][nj] += self.transition_prob[di+1][dj+1] * self.belief_state[i][j]
        self.belief_state = new_belief

    def update(self, observation):
        # This method updates the belief state based on the new observation.
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.belief_state[i][j] *= self.observation_prob[observation][i][j]
        self.belief_state /= np.sum(self.belief_state)

    def get_belief_state(self):
        return self.belief_state

# Example parameters
grid_size = 5
transition_prob = np.array([[0.1, 0.1, 0.1],
                            [0.1, 0.2, 0.1],
                            [0.1, 0.1, 0.1]])

observation_prob = {
    'A': np.random.rand(grid_size, grid_size),
    'B': np.random.rand(grid_size, grid_size)
}

hmm = RobotHMM(grid_size, transition_prob, observation_prob)
observations = ['A', 'B', 'A', 'B', 'A']

for obs in observations:
    hmm.predict()
    hmm.update(obs)
    print("Belief State after observation", obs)
    print(hmm.get_belief_state())
