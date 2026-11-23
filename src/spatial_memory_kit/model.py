"""Fixed-size recurrent comparison, isolated from all earlier Brain users."""
import copy
from typing import List
import torch
from .brain import Brain
from .reproducibility import seed_everything


ARCHITECTURE = dict(recurrent=True, horizons=(1, 5), hidden_size=128,
    action_modulation=True, memory_bias=4., history_horizons=(),
    prediction_mode='rollout', rollout_decoder_width=280, rollout_memory_bias=2.)
ARMS = ('separate_b2', 'coupled_b2', 'separate_b7', 'coupled_b7')


@torch.jit.script
def observation_sequence(x: torch.Tensor, previous_action: torch.Tensor, state: torch.Tensor,
                         wi: torch.Tensor, wh: torch.Tensor, bi: torch.Tensor, bh: torch.Tensor,
                         twi: torch.Tensor, twh: torch.Tensor, tbi: torch.Tensor, tbh: torch.Tensor,
                         coupled: bool) -> torch.Tensor:
    """Same fused ATen GRU-cell operation used by nn.GRUCell in PyTorch 2.5.1.

    Both arms use this execution path. The coupled arm computes a prior at each
    frame, but the all-zero previous-action sentinel discards it at the true
    episode start. Thus computed priors and causally adopted priors differ by
    one on full episodes; reset windows retain their actual preceding action.
    """
    states = torch.jit.annotate(List[torch.Tensor], [])
    for step in range(x.size(1)):
        if coupled:
            prior = torch._VF.gru_cell(previous_action[:, step], state, twi, twh, tbi, tbh)
            present = previous_action[:, step].ne(0).any(-1, keepdim=True)
            state = torch.where(present, prior, state)
        state = torch._VF.gru_cell(x[:, step], state, wi, wh, bi, bh)
        states.append(state)
    return torch.stack(states, 1)


class PriorCorrectionBrain(Brain):
    def __init__(self, observation_update='separate', **architecture):
        super().__init__(**architecture)
        if not self.recurrent or self.prediction_mode != 'rollout':
            raise ValueError('Comparison requires the recurrent rollout architecture')
        if observation_update not in ('separate', 'coupled'):
            raise ValueError('Unknown observation update')
        self.observation_update = observation_update

    def beliefs(self, image, body, previous_action, hidden=None, reset_each_step=False):
        b, t = image.shape[:2]
        encoded = self.encoder(image.reshape(b*t, 1, 16, 16)).reshape(b, t, -1)
        return self.beliefs_from_encoded(encoded, body, previous_action, hidden, reset_each_step)

    def beliefs_from_encoded(self, encoded, body, previous_action, hidden=None, reset_each_step=False):
        b, t = encoded.shape[:2]
        if t == 0:
            raise ValueError('Empty observation history')
        if reset_each_step:
            output, _ = self.beliefs_from_encoded(encoded.reshape(b*t, 1, -1),
                body.reshape(b*t, 1, -1), previous_action.reshape(b*t, 1, -1))
            return output.reshape(b, t, -1), None
        if hidden is not None and hidden.shape != (1, b, self.hidden_size):
            raise ValueError('Hidden shape must be [1,batch,hidden_size]')
        x = torch.cat((encoded, body, previous_action), -1)
        state = x.new_zeros(b, self.hidden_size) if hidden is None else hidden[0]
        c, p = self.core, self.transition
        output = observation_sequence(x, previous_action, state,
            c.weight_ih_l0, c.weight_hh_l0, c.bias_ih_l0, c.bias_hh_l0,
            p.weight_ih, p.weight_hh, p.bias_ih, p.bias_hh, self.observation_update == 'coupled')
        return output, output[:, -1][None]


def architecture(arm):
    update, bias = arm.split('_b')
    assert arm in ARMS
    return dict(ARCHITECTURE, observation_update=update, rollout_memory_bias=float(bias))


def fresh_family(seed, device='cpu'):
    """One random draw, four copies; b7 differs solely by float32 add_(5)."""
    seed_everything(seed)
    base = PriorCorrectionBrain(**architecture('separate_b2'))
    family = {}
    for arm in ARMS:
        model = copy.deepcopy(base)
        model.observation_update = arm.split('_b')[0]
        if arm.endswith('_b7'):
            with torch.no_grad():
                model.transition.bias_ih[128:256].add_(5.)
        family[arm] = model.to(device)
    return family
