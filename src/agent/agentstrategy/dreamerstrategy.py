from jax import numpy as jnp
from src.agent.agentstrategy.strategyinterface import StrategyInterface


class DreamerStrategy(StrategyInterface):
    def __init__(self):
        pass

    def update(self, old_state: jnp.ndarray, selected_action: int, reward: float, new_state: jnp.ndarray, done: bool):
        pass

    def action_policy(self, state: jnp.ndarray):
        pass

    def save(self):
        pass

    def load(self):
        pass

        # update actor
            actor_loss = self._compute_loss_actor(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.actor_optimizer.step()

            for p in self.world_param:
                p.requires_grad = True
            for p in self.value_model.parameters():
                p.requires_grad = True

            # update critic
            imag_beliefs = imag_beliefs.detach()
            imag_states = imag_states.detach()

            critic_loss = self._compute_loss_critic(imag_beliefs, imag_states, imag_ac_logps=imag_ac_logps)

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.args.grad_clip_norm, norm_type=2)
            self.value_optimizer.step()


def _compute_loss_critic(self, imag_beliefs, imag_states, imag_ac_logps=None):
    with torch.no_grad():
        # calculate the target with the target nn
        target_imag_values = bottle(self.target_value_model, (imag_beliefs, imag_states))
        imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))

        if self.args.pcont:
            pcont = bottle(self.pcont_model, (imag_beliefs, imag_states))
        else:
            pcont = self.args.discount * torch.ones_like(imag_rewards)

        if imag_ac_logps is not None:
            target_imag_values[1:] -= self.args.temp * imag_ac_logps

    returns = cal_returns(imag_rewards[:-1], target_imag_values[:-1], target_imag_values[-1], pcont[:-1],
                          lambda_=self.args.disclam)
    target_return = returns.detach()

    value_pred = bottle(self.value_model, (imag_beliefs, imag_states))[:-1]

    value_loss = F.mse_loss(value_pred, target_return, reduction="none").mean(dim=(0, 1))

    return value_loss

   def _compute_loss_actor(self, imag_beliefs, imag_states, imag_ac_logps=None):
        # reward and value prediction of imagined trajectories
        imag_rewards = bottle(self.reward_model, (imag_beliefs, imag_states))
        imag_values = bottle(self.value_model, (imag_beliefs, imag_states))

        with torch.no_grad():
            if self.args.pcont:
                pcont = bottle(self.pcont_model, (imag_beliefs, imag_states))
            else:
                pcont = self.args.discount * torch.ones_like(imag_rewards)
        pcont = pcont.detach()

        if imag_ac_logps is not None:
            imag_values[1:] -= self.args.temp * imag_ac_logps  # add entropy here

        returns = cal_returns(imag_rewards[:-1], imag_values[:-1], imag_values[-1], pcont[:-1],
                              lambda_=self.args.disclam)

        discount = torch.cumprod(torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0).detach()

        actor_loss = -torch.mean(discount * returns)
        return actor_loss