from types import SimpleNamespace

import torch
import pytest

from supermimic.imagination.objective import gae_targets, policy_objective


def test_terminal_transition_keeps_its_reward_and_blocks_future_values():
    rewards = torch.tensor([[.02], [.03], [.04]], requires_grad=True)
    values = torch.tensor([[5.], [6.], [1000.], [2000.]], requires_grad=True)
    done = torch.tensor([[0.], [1.], [0.]])
    targets = gae_targets(rewards, done, values, .99, 1.)
    torch.testing.assert_close(targets[1], rewards[1].detach())
    torch.testing.assert_close(targets[0], torch.tensor([.02 + .99 * .03]))
    assert not targets.requires_grad


def test_lambda_extremes_match_independent_discounted_sum_and_one_step_target():
    torch.manual_seed(7)
    rewards, values = torch.rand(6, 4), torch.rand(7, 4)
    done = torch.rand(6, 4)
    gamma = .99
    one_step = rewards + gamma * (1-done)*values[1:]
    torch.testing.assert_close(gae_targets(rewards, done, values, gamma, 0.), one_step)
    survival = torch.cat([torch.ones(1, 4), (1-done).cumprod(0)])
    discounted = gamma ** torch.arange(6)
    expected = (discounted[:, None]*survival[:-1]*rewards).sum(0) + gamma**6*survival[-1]*values[-1]
    torch.testing.assert_close(gae_targets(rewards, done, values, gamma, 1.)[0], expected)


def test_bounded_bootstrap_cannot_create_targets_above_the_true_upper_bound():
    torch.manual_seed(8)
    rewards = .1 - torch.rand(32, 64)
    values = 10. - 20*torch.rand(33, 64)
    targets = gae_targets(rewards, torch.rand(32, 64), values, .99, .95)
    assert targets.max() <= 10.00001
    fixed = gae_targets(torch.full((32, 1), .1), torch.zeros(32, 1), torch.full((33, 1), 10.), .99, .95)
    torch.testing.assert_close(fixed, torch.full_like(fixed, 10.))


def test_frozen_critic_preserves_state_gradient_and_targets_do_not_update_actor():
    actor = torch.nn.Linear(1, 1, bias=False)
    critic = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        actor.weight.fill_(2.)
        critic.weight.fill_(3.)
    critic.requires_grad_(False)
    state = actor(torch.ones(1, 1))
    rollout = SimpleNamespace(rewards=torch.zeros(1, 1), survival=torch.ones(2, 1))
    policy_objective(rollout, critic(state).flatten(), .99).backward()
    torch.testing.assert_close(actor.weight.grad, torch.tensor([[.99 * 3]]))
    assert critic.weight.grad is None
    critic.requires_grad_(True)
    actor.zero_grad(set_to_none=True)
    target = gae_targets(torch.ones(1, 1)*.1, torch.zeros(1, 1), torch.cat([state, state]), .99, .95)
    loss = (critic(state.detach())-target).square().mean()
    loss.backward()
    assert actor.weight.grad is None
    assert critic.weight.grad is not None


def test_value_and_survival_gradient_paths_sum_to_bootstrap_gradient():
    x = torch.tensor(.4, requires_grad=True)
    survival, value = torch.sigmoid(x), x*x + 3
    combined = torch.autograd.grad(survival*value, x, retain_graph=True)[0]
    value_path = torch.autograd.grad(survival.detach()*value, x, retain_graph=True)[0]
    survival_path = torch.autograd.grad(survival*value.detach(), x)[0]
    torch.testing.assert_close(combined, value_path + survival_path)


def test_learning_rate_schedule_preserves_separate_critic_setting():
    from supermimic.imagination.policy_trainer import PolicyTrainer
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    trainer.opt = SimpleNamespace(param_groups=[{"lr": 2e-4}])
    trainer.critic_opt = SimpleNamespace(param_groups=[{"lr": 5e-5}])
    trainer.cfg = SimpleNamespace(lr=2e-4, critic_lr=5e-5)
    trainer.set_lr(1e-4)
    assert trainer.opt.param_groups[0]["lr"] == 1e-4
    assert trainer.critic_opt.param_groups[0]["lr"] == pytest.approx(2.5e-5)
    trainer.cfg.critic_lr = trainer.cfg.lr
    trainer.set_lr(1e-4)
    assert trainer.critic_opt.param_groups[0]["lr"] == 1e-4
