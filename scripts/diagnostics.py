"""
Diagnostica completa del sistema di training Scopa AI.

Verifica:
1. Environment: reset, step, reward, done
2. Observation space: dimensioni, range, contenuto
3. Action masking: coerenza con mosse legali
4. Policy: forward pass, gradients
5. Rewards: distribuzione e correlazione con vittorie
6. Training loop: loss convergence
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import gymnasium as gym

from scopa.rl import ScopaEnv
from scopa.rl.policies import MaskableRecurrentPolicy
from scopa.game import Card, Suit
from scopa.config import OBSERVATION_DIM, ACTION_DIM, LSTM_CONFIG


def test_environment_basics():
    """Test 1: Verifica funzionamento base dell'environment."""
    print("\n" + "="*60)
    print("TEST 1: Environment Basics")
    print("="*60)
    
    env = ScopaEnv(opponent_mode="random")
    
    # Test reset
    obs, info = env.reset()
    assert obs.shape == (OBSERVATION_DIM,), f"Obs shape wrong: {obs.shape}"
    assert obs.min() >= 0 and obs.max() <= 1, f"Obs range wrong: [{obs.min()}, {obs.max()}]"
    print(f"✅ Reset OK: obs shape={obs.shape}, range=[{obs.min():.2f}, {obs.max():.2f}]")
    
    # Test action mask
    mask = env.action_masks()
    assert mask.shape == (ACTION_DIM,), f"Mask shape wrong: {mask.shape}"
    assert mask.sum() > 0, "No valid actions!"
    print(f"✅ Action mask OK: {mask.sum()} valid actions")
    
    # Test step with valid action
    valid_actions = np.where(mask)[0]
    action = valid_actions[0]
    obs2, reward, done, trunc, info = env.step(action)
    assert obs2.shape == (OBSERVATION_DIM,), f"Obs2 shape wrong"
    print(f"✅ Step OK: action={action}, reward={reward:.2f}, done={done}")
    
    # Play full game
    total_reward = 0
    steps = 0
    obs, _ = env.reset()
    while True:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            print(f"❌ No valid actions at step {steps}!")
            break
        action = np.random.choice(valid)
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if done or trunc:
            break
    
    print(f"✅ Full game: {steps} steps, total_reward={total_reward:.2f}")
    
    env.close()
    return True


def test_observation_content():
    """Test 2: Verifica contenuto dell'observation."""
    print("\n" + "="*60)
    print("TEST 2: Observation Content")
    print("="*60)
    
    env = ScopaEnv(opponent_mode="random")
    obs, _ = env.reset()
    
    # Scomposizione observation
    hand_obs = obs[0:40]
    table_obs = obs[40:80]
    captured_p0 = obs[80:120]
    captured_p1 = obs[120:160]
    opp_played = obs[160:200]
    selected = obs[200:240]
    deck_remaining = obs[240]
    stats = obs[241:252]
    last_capture = obs[252]
    capture_mode = obs[253]
    card_played_idx = obs[254]
    history = obs[255:295]
    starter = obs[295]
    
    print(f"Hand cards: {hand_obs.sum():.0f}")
    print(f"Table cards: {table_obs.sum():.0f}")
    print(f"Captured P0: {captured_p0.sum():.0f}")
    print(f"Captured P1: {captured_p1.sum():.0f}")
    print(f"Deck remaining: {deck_remaining:.2f} ({int(deck_remaining*40)} cards)")
    print(f"Starter player: {starter:.0f}")
    print(f"Capture mode (phase): {capture_mode:.0f}")
    
    # Verifica coerenza
    hand_cards = int(hand_obs.sum())
    assert hand_cards == 3, f"Expected 3 cards in hand, got {hand_cards}"
    print(f"✅ Hand has exactly 3 cards")
    
    table_cards = int(table_obs.sum())
    assert 1 <= table_cards <= 10, f"Table cards out of range: {table_cards}"
    print(f"✅ Table has {table_cards} cards (valid)")
    
    # Verifica che hand e action mask siano coerenti
    mask = env.action_masks()
    mask_sum = mask.sum()
    
    # In fase 0, le azioni valide dovrebbero corrispondere alle carte in mano
    if capture_mode == 0:
        hand_indices = np.where(hand_obs > 0)[0]
        mask_indices = np.where(mask)[0]
        if not np.array_equal(sorted(hand_indices), sorted(mask_indices)):
            print(f"⚠️ Hand indices {hand_indices} != Mask indices {mask_indices}")
        else:
            print(f"✅ Action mask matches hand cards")
    
    env.close()
    return True


def test_action_masking_consistency():
    """Test 3: Verifica consistenza action masking."""
    print("\n" + "="*60)
    print("TEST 3: Action Masking Consistency")
    print("="*60)
    
    env = ScopaEnv(opponent_mode="random")
    
    inconsistencies = 0
    for game in range(10):
        obs, _ = env.reset()
        
        for step in range(50):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            
            if len(valid) == 0:
                print(f"❌ Game {game}, step {step}: No valid actions!")
                inconsistencies += 1
                break
            
            # Prova azione valida
            action = np.random.choice(valid)
            try:
                obs, reward, done, trunc, info = env.step(action)
            except Exception as e:
                print(f"❌ Step failed: {e}")
                inconsistencies += 1
                break
            
            if done or trunc:
                break
    
    if inconsistencies == 0:
        print(f"✅ All 10 games completed without masking issues")
    else:
        print(f"❌ {inconsistencies} inconsistencies found")
    
    env.close()
    return inconsistencies == 0


def test_reward_distribution():
    """Test 4: Analizza distribuzione dei reward."""
    print("\n" + "="*60)
    print("TEST 4: Reward Distribution")
    print("="*60)
    
    env = ScopaEnv(opponent_mode="random")
    
    final_rewards = []
    step_rewards = []
    wins = 0
    
    for game in range(100):
        obs, _ = env.reset()
        game_reward = 0
        
        while True:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = np.random.choice(valid)
            obs, reward, done, trunc, info = env.step(action)
            
            if reward != 0:
                step_rewards.append(reward)
            game_reward += reward
            
            if done or trunc:
                final_rewards.append(game_reward)
                # Determina vittoria
                scores = env.engine.calculate_score()
                if scores[0] > scores[1]:
                    wins += 1
                break
    
    final_rewards = np.array(final_rewards)
    step_rewards = np.array(step_rewards) if step_rewards else np.array([0])
    
    print(f"Games played: 100")
    print(f"Random win rate: {wins}%")
    print(f"Final rewards: mean={final_rewards.mean():.2f}, std={final_rewards.std():.2f}")
    print(f"  range: [{final_rewards.min():.0f}, {final_rewards.max():.0f}]")
    print(f"Step rewards: mean={step_rewards.mean():.2f}, std={step_rewards.std():.2f}")
    
    # Win rate dovrebbe essere ~50% vs random
    if 40 <= wins <= 60:
        print(f"✅ Random vs Random win rate ~50% as expected")
    else:
        print(f"⚠️ Win rate {wins}% deviates from expected 50%")
    
    env.close()
    return True


def test_policy_forward():
    """Test 5: Verifica forward pass della policy."""
    print("\n" + "="*60)
    print("TEST 5: Policy Forward Pass")
    print("="*60)
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=(OBSERVATION_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(ACTION_DIM)
    
    policy = MaskableRecurrentPolicy(
        obs_space, act_space,
        features_dim=LSTM_CONFIG["features_dim"],
        lstm_hidden_size=LSTM_CONFIG["lstm_hidden_size"],
        lstm_num_layers=LSTM_CONFIG["lstm_num_layers"],
    )
    
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Test single observation
    obs = torch.randn(1, OBSERVATION_DIM)
    lstm_states = policy.get_initial_state(1)
    episode_starts = torch.zeros(1, dtype=torch.bool)
    action_masks = torch.ones(1, ACTION_DIM, dtype=torch.bool)
    action_masks[0, 10:] = False  # Only first 10 actions valid
    
    actions, values, log_probs, new_states = policy.forward(
        obs, lstm_states, episode_starts, action_masks
    )
    
    print(f"Actions shape: {actions.shape}, value: {actions.item()}")
    print(f"Values shape: {values.shape}, value: {values.item():.2f}")
    print(f"Log probs shape: {log_probs.shape}, value: {log_probs.item():.2f}")
    
    # Action dovrebbe essere < 10 (masked)
    assert actions.item() < 10, f"Action {actions.item()} violates mask!"
    print(f"✅ Action masking works correctly")
    
    # Test batch
    batch_size = 4
    obs_batch = torch.randn(batch_size, OBSERVATION_DIM)
    lstm_states = policy.get_initial_state(batch_size)
    episode_starts = torch.zeros(batch_size, dtype=torch.bool)
    action_masks = torch.ones(batch_size, ACTION_DIM, dtype=torch.bool)
    
    actions, values, log_probs, new_states = policy.forward(
        obs_batch, lstm_states, episode_starts, action_masks
    )
    
    print(f"✅ Batch forward OK: actions={actions.tolist()}")
    
    return True


def test_policy_gradients():
    """Test 6: Verifica che i gradienti fluiscono correttamente."""
    print("\n" + "="*60)
    print("TEST 6: Policy Gradients")
    print("="*60)
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=(OBSERVATION_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(ACTION_DIM)
    
    policy = MaskableRecurrentPolicy(obs_space, act_space)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Forward pass
    batch_size = 8
    obs = torch.randn(batch_size, OBSERVATION_DIM)
    actions = torch.randint(0, ACTION_DIM, (batch_size,))
    lstm_states = policy.get_initial_state(batch_size)
    episode_starts = torch.zeros(batch_size, dtype=torch.bool)
    action_masks = torch.ones(batch_size, ACTION_DIM, dtype=torch.bool)
    
    # Evaluate actions
    values, log_probs, entropy = policy.evaluate_actions(
        obs, actions, lstm_states, episode_starts, action_masks
    )
    
    # Fake advantages and returns
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    
    # Compute loss
    policy_loss = -(advantages * log_probs).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy_loss = -entropy.mean()
    
    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
    
    print(f"Policy loss: {policy_loss.item():.4f}")
    print(f"Value loss: {value_loss.item():.4f}")
    print(f"Entropy: {entropy.mean().item():.4f}")
    print(f"Total loss: {loss.item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    print(f"\nGradient norms (sample):")
    for name, norm in grad_norms[:5]:
        print(f"  {name}: {norm:.4f}")
    
    # All gradients should be non-zero
    zero_grads = [n for n, norm in grad_norms if norm == 0]
    if zero_grads:
        print(f"⚠️ Zero gradients in: {zero_grads}")
    else:
        print(f"✅ All {len(grad_norms)} parameters have non-zero gradients")
    
    # Step optimizer
    optimizer.step()
    print(f"✅ Optimizer step completed")
    
    return True


def test_training_makes_sense():
    """Test 7: Verifica che il training migliora policy."""
    print("\n" + "="*60)
    print("TEST 7: Training Sanity Check")
    print("="*60)
    
    obs_space = gym.spaces.Box(low=0, high=1, shape=(OBSERVATION_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(ACTION_DIM)
    
    policy = MaskableRecurrentPolicy(obs_space, act_space)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    # Simula training: policy dovrebbe imparare a preferire azione 0 dato target
    print("Training policy to prefer action 0...")
    
    losses = []
    for step in range(100):
        batch_size = 32
        obs = torch.randn(batch_size, OBSERVATION_DIM)
        target_actions = torch.zeros(batch_size, dtype=torch.long)
        lstm_states = policy.get_initial_state(batch_size)
        episode_starts = torch.zeros(batch_size, dtype=torch.bool)
        action_masks = torch.ones(batch_size, ACTION_DIM, dtype=torch.bool)
        
        # Evaluate
        values, log_probs, entropy = policy.evaluate_actions(
            obs, target_actions, lstm_states, episode_starts, action_masks
        )
        
        # Loss: maximize log prob of action 0
        loss = -log_probs.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    initial_loss = np.mean(losses[:10])
    final_loss = np.mean(losses[-10:])
    
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    
    if final_loss < initial_loss:
        print(f"✅ Loss decreased: policy is learning!")
    else:
        print(f"❌ Loss did not decrease: potential issue")
    
    # Test if policy now prefers action 0
    policy.eval()
    with torch.no_grad():
        obs_test = torch.randn(100, OBSERVATION_DIM)
        lstm_states = policy.get_initial_state(100)
        episode_starts = torch.zeros(100, dtype=torch.bool)
        action_masks = torch.ones(100, ACTION_DIM, dtype=torch.bool)
        
        actions, _, _, _ = policy.forward(
            obs_test, lstm_states, episode_starts, action_masks, deterministic=True
        )
    
    action_0_rate = (actions == 0).float().mean().item()
    print(f"Action 0 selection rate: {action_0_rate:.1%}")
    
    if action_0_rate > 0.5:
        print(f"✅ Policy learned to prefer action 0")
    else:
        print(f"⚠️ Policy has {action_0_rate:.1%} rate for action 0")
    
    return final_loss < initial_loss


def test_environment_vs_random():
    """Test 8: Benchmark random vs random per baseline."""
    print("\n" + "="*60)
    print("TEST 8: Random vs Random Baseline")
    print("="*60)
    
    # Run many games with both players random
    wins_p0 = 0
    total_score_diff = 0
    
    for _ in range(500):
        env = ScopaEnv(opponent_mode="random")
        obs, _ = env.reset()
        
        while True:
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = np.random.choice(valid)
            obs, reward, done, trunc, info = env.step(action)
            
            if done or trunc:
                scores = env.engine.calculate_score()
                if scores[0] > scores[1]:
                    wins_p0 += 1
                total_score_diff += scores[0] - scores[1]
                break
        
        env.close()
    
    win_rate = wins_p0 / 500
    avg_diff = total_score_diff / 500
    
    print(f"Games: 500")
    print(f"P0 win rate: {win_rate:.1%}")
    print(f"Avg score diff: {avg_diff:+.2f}")
    
    # Dovrebbe essere ~50% dato che entrambi sono random
    if 0.45 <= win_rate <= 0.55:
        print(f"✅ Win rate ~50% as expected for random vs random")
    else:
        print(f"⚠️ Win rate {win_rate:.1%} deviates significantly from 50%")
    
    return True


def main():
    print("\n" + "="*60)
    print("🔬 SCOPA AI DIAGNOSTIC TESTS")
    print("="*60)
    
    tests = [
        ("Environment Basics", test_environment_basics),
        ("Observation Content", test_observation_content),
        ("Action Masking", test_action_masking_consistency),
        ("Reward Distribution", test_reward_distribution),
        ("Policy Forward", test_policy_forward),
        ("Policy Gradients", test_policy_gradients),
        ("Training Sanity", test_training_makes_sense),
        ("Random Baseline", test_environment_vs_random),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Review output above for details.")


if __name__ == "__main__":
    main()
