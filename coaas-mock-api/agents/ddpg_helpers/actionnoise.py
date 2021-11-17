import numpy as np 

class ActionNoise():
    # Based on OpenAI Baseline
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, start_noise=None):
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.theta = theta
        self.start_noise = start_noise
        self.prev_noise = self.start_noise if self.start_noise is not None else np.zeros_like(self.mu)
    
    def __call__(self):
        noise = self.prev_noise + (self.theta * (self.mu - self.prev_noise) * self.dt) + (self.sigma * 
                np.sqrt(self.dt) * np.random.normal(size=self.mu.shape))
        self.prev_noise = noise

        return noise