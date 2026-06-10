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
    # 0.8^(100k/5000) -> final LR ~1.2e-5.  The previous 0.85/2000 schedule
    # decayed to 2.9e-7 by step 100k, freezing the last half of training.
    optim.decay_rate = 0.8
    optim.decay_steps = 5000

    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size = 1024
    training.xi_min = 0.0
    training.xi_max = 1.0
    training.eta_min = 2e-3
    training.eta_max = 0.995
    # Entrance/wall-clustered collocation: fraction of each batch drawn with
    # xi ~ u^s_power (toward inlet) and eta ~ wall-clustered (toward eta_max),
    # to resolve the thin inlet shear layer of the true-plug entrance problem.
    training.entrance_cluster_frac = 0.4
    training.cluster_s_power = 2.0
    training.cluster_e_power = 1.5

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({
        "res_c": 1.0, "res_x": 1.0, "res_r": 1.0, "res_k": 1.0, "res_w": 1.0,
        "sym": 1.0, "bc_inlet": 10.0, "bc_outlet": 1.0, "pgauge": 1.0,
        "anchor": 1.0, "anchor_k": 10.0, "wall_shear": 10.0, "mass": 100.0,
    })
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000
    # Guards against grad-norm weight explosion (observed: w_res_r -> 1e10 once
    # res_r hit the float32 noise floor, drowning the unconverged res_c term).
    weighting.max_weight = 1.0e3
    weighting.loss_floor = 1.0e-8
    # Per-key weight floors: grad-norm de-prioritised mass (762 -> 69) while
    # the bulk velocity drifted -1.5% along the pipe.
    weighting.min_weights = ml_collections.ConfigDict({"mass": 100.0})
    # Anchor annealing: the Reichardt U anchor and the approximate k anchor are
    # empirical, not exact k-omega solutions; once the turbulent branch is
    # established they fight the momentum residual (grad cosine anchor/res_x
    # ~ -0.75 at convergence).  Decay them in the second half of training so
    # the PDE settles on its own fully-developed solution.  wall_shear and
    # mass stay: they are exact physics and together forbid laminar collapse
    # (tau_w+=1 with bulk 14.28 is incompatible with a laminar profile).
    weighting.anneal = ml_collections.ConfigDict({
        "keys": ("anchor", "anchor_k"),
        "start_step": 40000,
        "rate": 0.5,
        "period": 5000,
    })

    config.physics = physics = ml_collections.ConfigDict()
    physics.re_tau = 550.0
    # Aspect ratio L/R: pipe length / pipe radius.  Turbulent entrance length
    # is Le/D ~ 1.6 Re_D^(1/4) ~ 14 D, so AR=32 (L=16D) lets a true plug inlet
    # develop naturally inside the domain (AR=20 was shorter than Le).
    physics.aspect_ratio = 32.0
    # Inlet: near-plug profile U_plug*tanh(y+/delta_inlet).  delta_inlet=6 puts
    # the inlet wall shear at U_plug/delta ~ 2.5 (>1, decaying downstream like a
    # real entrance flow) while still regularising the plug/no-slip corner.
    # The plug level is set internally so the inlet mass flux equals the
    # fully-developed (Reichardt) bulk velocity at this Re_tau; an explicit
    # u_inlet would over-specify the problem in wall units.
    physics.delta_inlet = 6.0
    # Inlet turbulent kinetic energy scale in wall units (peak k+ of the inlet
    # profile; the profile decays to 0 at the wall like the velocity squared).
    physics.k_inlet = 1.0

    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500
    logging.log_losses = True
    logging.log_weights = True
    logging.log_errors = True

    config.input_dim = 2
    config.seed = 0
    return config
