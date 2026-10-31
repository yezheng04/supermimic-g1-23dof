"""Small recorder interface for mjlab 1.2; owned locally, no site-packages patches."""
from dataclasses import dataclass, field


@dataclass
class RecorderTermCfg:
    func: type | None = None
    params: dict = field(default_factory=dict)


class RecorderTerm:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env


class RecorderManager:
    def __init__(self, cfgs, env):
        self.terms = {name: cfg.func(cfg, env) for name, cfg in cfgs.items()}

    def get_term(self, name):
        return self.terms[name]

    def call(self, hook, *args):
        for term in self.terms.values():
            getattr(term, hook)(*args)
