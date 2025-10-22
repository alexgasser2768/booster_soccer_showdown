import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ======================================================
# 1. Define equivalent PyTorch network
# ======================================================
class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activate_final=True, layer_norm=True):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        
        last_dim = input_dim
        for h in hidden_layers:
            self.hidden_layers.append(nn.Linear(last_dim, h))
            if layer_norm:
                self.layer_norms.append(nn.LayerNorm(h, eps=1e-6))
            last_dim = h

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if i + 1 < len(self.hidden_layers) or self.activate_final:
                x = F.relu(x)
                if self.layer_norm:
                    x = self.layer_norms[i](x)
        return x


class TorchGCActor(nn.Module):
    def __init__(self, obs_dim, hidden_layers, action_dim, const_std=False):
        super().__init__()
        self.const_std = const_std
        self.actor_net = TorchMLP(obs_dim, hidden_layers, activate_final=True, layer_norm=True)
        last_hidden = hidden_layers[-1]
        self.mean_net = nn.Linear(last_hidden, action_dim)
        self.log_std_net = nn.Linear(last_hidden, action_dim)
    
    def forward(self, obs, temperature=0.0):
        x = obs
        feat = self.actor_net(x)
        mean = self.mean_net(feat)
        log_std = self.log_std_net(feat)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std) * temperature
        return mean, std
    
    @torch.no_grad()
    def get_actions(self, observation, temperature=1.0):
        """
        Equivalent to JAX get_actions()
        - Produces a Gaussian policy
        - Samples an action
        - Clips to [-1, 1]
        """
        mean, std = self.forward(observation, temperature=temperature)
        dist = Normal(mean, std)
        actions = dist.rsample()  # reparameterized sample
        actions = torch.clamp(actions, -1.0, 1.0)
        return actions

# ======================================================
# 2. Load JAX weights (.pkl)
# ======================================================
pkl_path = "/Users/shaswatgarg/Documents/Job/ArenaX/Development/booster_soccer_showdown/imitation_learning/exp/booster/Debug/`cobot_pick_place_20251021-212615_bc/params_1000000.pkl"
with open(pkl_path, "rb") as f:
    jax_params = pickle.load(f)

params = jax_params["agent"]["network"]["params"]["modules_actor"]

# ======================================================
# 3. Initialize and load weights into PyTorch
# ======================================================
torch_model = TorchGCActor(obs_dim=51, hidden_layers=[256, 256], action_dim=12, const_std=False)

def load_dense(torch_layer, jax_layer):
    torch_layer.weight.data = torch.tensor(jax_layer["kernel"]).T
    torch_layer.bias.data = torch.tensor(jax_layer["bias"])

def load_layernorm(torch_ln, jax_ln):
    torch_ln.bias.data = torch.tensor(jax_ln["bias"])
    torch_ln.weight.data = torch.tensor(jax_ln["scale"])

# Load MLP (actor_net)
actor_net = params["actor_net"]
load_dense(torch_model.actor_net.hidden_layers[0], actor_net["Dense_0"])
load_dense(torch_model.actor_net.hidden_layers[1], actor_net["Dense_1"])
load_layernorm(torch_model.actor_net.layer_norms[0], actor_net["LayerNorm_0"])
load_layernorm(torch_model.actor_net.layer_norms[1], actor_net["LayerNorm_1"])

# Load heads
load_dense(torch_model.mean_net, params["mean_net"])
load_dense(torch_model.log_std_net, params["log_std_net"])

# ======================================================
# 4. Test forward pass
# ======================================================
obs = torch.randn(1, 51)
mean, std = torch_model(obs)
print("mean shape:", mean.shape, "std shape:", std.shape)

# ======================================================
# 5. Save TorchScript version
# ======================================================
scripted_model = torch.jit.trace(torch_model, (torch.randn(1, 51),))
torch.jit.save(scripted_model, "converted_gc_actor.pt")

print("✅ TorchScript model saved as converted_gc_actor.pt")
