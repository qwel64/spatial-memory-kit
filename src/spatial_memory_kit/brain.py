import torch
from torch import nn
from torch.nn import functional as F


class ActionModulatedHead(nn.Module):
    """Learn the motor-conditioned visual transform; no hand-coded geometry."""
    def __init__(self,hidden_size,horizon):
        super().__init__()
        self.state=nn.Linear(hidden_size,192)
        self.motor=nn.Linear(4*horizon,384)
        self.output=nn.Linear(192,261)
        self.hidden_size=hidden_size

    def forward(self,x):
        state=self.state(x[...,:self.hidden_size])
        scale,bias=self.motor(x[...,self.hidden_size:]).chunk(2,-1)
        return self.output(F.silu(state*(1+.5*scale.tanh())+bias))


class Brain(nn.Module):
    """Belief h_t uses o_0..o_t, a_0..a_(t-1); no future observation.

    Prediction heads can condition on a planned future action sequence.
    Those future actions NEVER enter h_t or the spatial probes.
    """
    def __init__(self, recurrent=True, horizons=(1,), hidden_size=128, action_modulation=False,memory_bias=0.,history_horizons=(),prediction_mode='direct',rollout_decoder_width=280,rollout_memory_bias=2.):
        super().__init__()
        self.recurrent = recurrent
        self.horizons = tuple(horizons)
        self.hidden_size = hidden_size
        self.history_horizons=tuple(history_horizons)
        if prediction_mode not in ('direct','rollout'):raise ValueError('Unknown prediction mode')
        self.prediction_mode=prediction_mode
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(), nn.Flatten(),
            nn.Linear(32*4*4, 64), nn.SiLU())
        self.core = (nn.GRU(64+5+4, hidden_size, batch_first=True) if recurrent else
                     nn.Sequential(nn.Linear(73, 384), nn.SiLU(), nn.Linear(384, hidden_size), nn.Tanh()))
        if recurrent and memory_bias:
            # A generic longer initial timescale, identical for trained and frozen GRU controls.
            with torch.no_grad():self.core.bias_ih_l0[hidden_size:2*hidden_size].add_(memory_bias)
        self.heads = nn.ModuleDict({str(k): (ActionModulatedHead(hidden_size,k) if action_modulation
            else nn.Sequential(nn.Linear(hidden_size+4*k, 192), nn.SiLU(), nn.Linear(192, 261))) for k in horizons} if prediction_mode=='direct' else {})
        if prediction_mode=='rollout':
            # One learned action-only transition, shared across all future steps.
            # The state has no prescribed position/angle components or geometry.
            self.transition=nn.GRUCell(4,hidden_size)
            with torch.no_grad():self.transition.bias_ih[hidden_size:2*hidden_size].add_(rollout_memory_bias)
            self.rollout_decoder=nn.Sequential(nn.Linear(hidden_size,rollout_decoder_width),nn.SiLU(),nn.Linear(rollout_decoder_width,261))
        self.history_heads=nn.ModuleDict({str(k):nn.Sequential(nn.Linear(hidden_size,192),nn.SiLU(),nn.Linear(192,261))
            for k in self.history_horizons})

    def beliefs(self, image, body, previous_action, hidden=None, reset_each_step=False):
        b, t = image.shape[:2]
        encoded = self.encoder(image.reshape(b*t, 1, 16, 16)).reshape(b, t, -1)
        x = torch.cat([encoded, body, previous_action], -1)
        if not self.recurrent:
            return self.core(x), None
        if reset_each_step:
            y, _ = self.core(x.reshape(b*t, 1, -1))
            return y.reshape(b, t, -1), None
        return self.core(x, hidden)

    def predict(self, belief, action_sequence, horizon=1):
        if self.prediction_mode=='rollout':
            if horizon not in self.horizons or action_sequence.shape[-1]!=horizon:
                raise ValueError('Prediction horizon does not match trained horizon and action sequence')
            return self.decode(self.rollout_decoder(self.imagine(belief,action_sequence)))
        x = torch.cat([belief, F.one_hot(action_sequence.long(), 4).float().flatten(-2)], -1)
        y = self.heads[str(horizon)](x)
        return self.decode(y)

    def imagine(self,belief,action_sequence):
        """Advance a copy of h_t using actions only; do not assimilate observations."""
        if self.prediction_mode!='rollout':raise ValueError('Imagine requires a rollout predictor')
        if belief.shape[:-1]!=action_sequence.shape[:-1]:raise ValueError('Belief and action batch shapes differ')
        state=belief.reshape(-1,self.hidden_size)
        for action in action_sequence.unbind(-1):
            state=self.transition(F.one_hot(action.long(),4).float().reshape(-1,4),state)
        return state.reshape(belief.shape)

    def reconstruct(self,belief,lag):
        return self.decode(self.history_heads[str(lag)](belief))

    @staticmethod
    def decode(y):
        return {'image': y[..., :256].sigmoid().reshape(*y.shape[:-1], 1, 16, 16),
                'contact_logits': y[..., 256:258], 'motion': y[..., 258:260].tanh(),
                'energy': y[..., 260:261].sigmoid()}


def prediction_loss(prediction, image, body):
    image_mse = F.mse_loss(prediction['image'], image)
    contact_bce = F.binary_cross_entropy_with_logits(prediction['contact_logits'], body[..., :2])
    motion_mse = F.mse_loss(prediction['motion'], body[..., 2:4])
    energy_mse = F.mse_loss(prediction['energy'], body[..., 4:5])
    loss = image_mse + .03*contact_bce + .02*motion_mse + .01*energy_mse
    return loss, {'image_mse': image_mse, 'contact_bce': contact_bce,
                  'motion_mse': motion_mse, 'energy_mse': energy_mse}
