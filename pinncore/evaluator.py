"""Base evaluator (mirrors jaxpi/evaluator.py).

Collects per-iteration scalars into a dict that train.py logs (to W&B or stdout).
A problem-specific evaluator subclasses this and adds error metrics / fields.
"""


class BaseEvaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.log_dict = {}

    def log_losses(self, params, batch):
        ld = self.model.losses(params, batch)
        for k, v in ld.items():
            self.log_dict[f"loss_{k}"] = float(v)

    def log_weights(self, state):
        for k, v in state.weights.items():
            self.log_dict[f"w_{k}"] = float(v)

    def __call__(self, state, batch):
        self.log_dict = {}
        if self.config.logging.log_losses:
            self.log_losses(state.params, batch)
        if self.config.logging.log_weights:
            self.log_weights(state)
        return self.log_dict
