"""mjlab viewer interface without rsl_rl or duplicate observation-history updates."""


class ViewerEnv:
    def __init__(self, env):
        self.env = env
        self.env.reset()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_observations(self):
        return self.env.obs_buf
