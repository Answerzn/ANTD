import torch
import tqdm
import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def compute_alpha(beta, t):
    """
    Compute continuous alpha_t (cumulative product of 1-beta) indexed by t.

    If `beta` is None, a default linear schedule is created.

    Args:
        beta: 1-D tensor of shape (T,) or None
        t: long/int tensor of timesteps (can be float but will be rounded by callers when discrete indexes are required)

    Returns:
        a: tensor shaped (len(t), 1, 1, 1) with alpha_{t}
    """
    if beta is None:
        # choose a reasonably large default T so indexing won't often go out of range
        beta_np = get_beta_schedule('linear', beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000)
        beta = torch.from_numpy(beta_np).float().to(t.device)
    else:
        # accept beta as numpy array or torch tensor
        if isinstance(beta, np.ndarray):
            beta = torch.from_numpy(beta).float().to(t.device)
        else:
            beta = beta.to(t.device).float()

    beta = torch.cat([torch.zeros(1, device=beta.device, dtype=beta.dtype), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def time_step_schedule(x0, t_pred, T=1000, beta=None):
    """
    Build a continuous-time starting noisy image and a discrete schedule from a predicted time.

    Motivation:
    - `t_pred` is a predictor output in (0,1) (sigmoid). We map it to [0, T-1].
    - To allow gradients to flow through the timestep, we compute alpha(t) by linear interpolation
      between the precomputed alpha_cumprod values (derived from `beta`). That gives a continuous
      alpha(t) which is differentiable w.r.t. t_pred (through the interpolation weights).

    Returns:
        x_start: Tensor shaped like x0, the noisy image at predicted continuous timestep
        seq: a Python list of integer timesteps to use as the sampling schedule (from start down to 0)

    Notes:
    - `generalized_steps` expects a global sequence of integer timesteps. We produce a simple
      descending integer schedule starting from the rounded mean predicted timestep.
    - If your training requires per-sample heterogeneous schedules, `generalized_steps` must be
      adapted. This helper keeps a single schedule (same for the batch) and a per-sample continuous
      start image so the network can still learn t_pred.
    """
    # x0: clean image tensor (N,C,H,W)
    # noise: noise tensor same shape
    # t_pred: predicted normalized time in (0,1), shape (N,1) or (N,)
    if isinstance(t_pred, torch.Tensor):
        t_cont = t_pred.view(-1).to(x0.device)
    else:
        t_cont = torch.tensor(t_pred, device=x0.device).view(-1)

    # map normalized [0,1] -> [0, T-1]
    t_cont = T * t_cont - 1

    # prepare alpha_cumprod table
    # build beta table as a torch tensor on the correct device
    if beta is None:
        beta_np = get_beta_schedule('linear', beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=T)
        beta_tbl = torch.from_numpy(beta_np).float().to(x0.device)
    else:
        if isinstance(beta, np.ndarray):
            beta_tbl = torch.from_numpy(beta).float().to(x0.device)
        else:
            beta_tbl = beta.to(x0.device).float()

    alpha_cum = (1 - beta_tbl).cumprod(dim=0)  # shape (T,)

    # For continuous t we do linear interpolation between floor and ceil alphas
    t_floor = torch.floor(t_cont).long().clamp(min=0, max=T - 1)
    t_ceil = torch.ceil(t_cont).long().clamp(min=0, max=T - 1)
    frac = (t_cont - t_floor.float()).unsqueeze(1)

    alpha_floor = alpha_cum.index_select(0, t_floor).unsqueeze(-1).unsqueeze(-1)
    alpha_ceil = alpha_cum.index_select(0, t_ceil).unsqueeze(-1).unsqueeze(-1)

    # alpha_t is continuous interpolation
    alpha_t = alpha_floor * (1 - frac).view(-1, 1, 1) + alpha_ceil * frac.view(-1, 1, 1)

    # shapes: (N,1,1) -> broadcast for image channels
    sqrt_alpha = alpha_t.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = (1 - alpha_t).sqrt().view(-1, 1, 1, 1)

    # build starting noisy image per-sample
    # x_start = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


    x_start = sqrt_alpha * x0

    return x_start, t_cont, alpha_t 


def generalized_steps(x, seq, model, b, eta=0., seq_next=None):
    with torch.no_grad():
        n = x.size(0)
        if seq_next == None:
            seq_next = [-1] + list(seq[:-1])

        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            #print(torch.ones(n).device,i.device)
            t = (torch.ones(n).to(x.device) * i).to(x.device)
            next_t = (torch.ones(n).to(x.device) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            z=torch.randn_like(x0_t)
            xt_next = at_next.sqrt() * x0_t + c1 * z + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


def precompute_ratio(betas):
    num_diffusion_timesteps = len(betas)
    t = torch.arange(num_diffusion_timesteps)
    a = compute_alpha(betas, t)
    ratio = torch.sqrt((1 - a) / a)
    return ratio

def find_best_t(ratio, sigma):
    abs_diff = torch.abs(ratio - sigma)
    best_t = torch.argmin(abs_diff).item()
    return best_t+1


def find_N(sigma=2*10/255):
    betas = get_beta_schedule(beta_schedule='linear',
                              beta_start=0.0001,
                              beta_end=0.02,
                              num_diffusion_timesteps=1000)
    betas = torch.from_numpy(betas).float()
    ratio = precompute_ratio(betas)
    # print(ratio)
    best_t = find_best_t(ratio, sigma)
    
    # for i in [15,25,50,100,150,200,250]:
    #     print(find_best_t(ratio, i*2/255))
    return best_t