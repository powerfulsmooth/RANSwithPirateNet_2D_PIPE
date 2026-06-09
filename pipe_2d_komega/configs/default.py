"""Default config for 2D developing turbulent pipe flow (standard k-omega)."""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.mode = "train"

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Pipe2D-KOmega"
    wandb.name = "default"
    wandb.mode = "disabled"

    # PirateNet — input (xi, eta), xi NOT periodic
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 4
    arch.hidden_dim = 128
    arch.out_dim = 5                    # U+, V+, p+, k+, omega+
    arch.activation = "tanh"
    arch.nonlinearity = 0.0
    arch.periodicity = None             # no periodicity in xi
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 5.0, "embed_dim": 128})
    arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": 0.5, "stddev": 0.1})

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.85
    optim.decay_steps = 2000

    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size = 1024
    training.xi_min = 0.0
    training.xi_max = 1.0
    training.eta_min = 2e-3
    training.eta_max = 0.995

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({
        "res_c": 1.0, "res_x": 1.0, "res_r": 1.0, "res_k": 1.0, "res_w": 1.0,
        "sym": 1.0, "bc_inlet": 10.0, "bc_outlet": 1.0, "pgauge": 1.0, "anchor": 1.0,
    })
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    config.physics = physics = ml_collections.ConfigDict()
    physics.re_tau = 550.0
    # Aspect ratio L/R: pipe length / pipe radius.
    # L+ = AR * Re_tau wall units.  AR=20 -> L+ = 11000 (moderately long pipe).
    physics.aspect_ratio = 20.0
    # Plug inlet velocity in wall units.
    # For Re_tau=550, turbulent bulk ~18-20 u_tau; use flat plug profile.
    physics.u_inlet = 18.0
    # Inlet turbulent kinetic energy in wall units (k+ = 3/2*(I*U+)^2, I~5%).
    physics.k_inlet = 1.0

    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.log_losses = True
    logging.log_weights = True
    logging.log_errors = True

    config.input_dim = 2
    config.seed = 0
    return config
