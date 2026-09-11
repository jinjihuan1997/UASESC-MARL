"""Audit online information used by the frozen myopic baseline; no training."""
import os
import sys
import copy
import ast
import hashlib
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parent / '2026-09-10_resource_selector_training'
sys.path.insert(0, str(ROOT))
import helpers as h
import rule_tools as rules
import numpy as np
import torch


class CurrentInstruction:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, index):
        assert isinstance(index, int) and index == self.env.step_index, 'Attempted non-current instruction read'
        return self.env.instructions[index]


class ParameterView:
    allowed = {'beta_sat_lower_bound', 'delta_T', 'backhaul_availability', 'B_uav_sut', 'B_sut_sat',
               'A_max', 'Q_min_eval', 'Q_max', 'reward_weights_by_instruction', 'A_limit_by_instruction',
               'constraint_penalty_A_by_instruction', 'n_ds', 'reward_load_ref', 'aoi_reward_ref',
               'aoi_tail_threshold', 'aoi_mean_weight', 'aoi_max_weight', 'aoi_tail_weight',
               'reward_load_scale', 'eta_recv_aoi_bonus'}

    def __init__(self, source, reads):
        self.source, self.reads = source, reads

    def __getattr__(self, name):
        assert name in self.allowed, 'Unapproved parameter read: ' + name
        self.reads.add('p.' + name)
        return getattr(self.source, name)


class CurrentStateView:
    allowed = {'U', 'K', 'M', 'count', 'n_agents', 'device', 'dtype', 'beta', 'q', 'tau', 'aoi',
               'step_index', 'quality', 'load', 'limits', 'requirements', 'buckets'}

    def __init__(self, env, reads):
        self.env, self.reads = env, reads

    def __getattr__(self, name):
        self.reads.add(name)
        if name == 'p':
            return ParameterView(self.env.p, self.reads)
        if name == 'instructions':
            return CurrentInstruction(self.env)
        assert name in self.allowed, 'Non-current/private environment access: ' + name
        return getattr(self.env, name)

    def context(self):
        return type(self.env).context(self)


def evaluate_candidates(env):
    actions = [h.rule_actions(env, name) for name in h.RULES]
    predicted = torch.stack([rules.predict_reward(env, action) for action in actions])
    return predicted, predicted.argmax(0)


def immutable_fingerprint(env):
    out = {}
    for name, value in vars(env).items():
        if torch.is_tensor(value):
            out[name] = hashlib.sha256(h.arr(value).tobytes()).hexdigest()
        elif isinstance(value, (bool, int, float, str)):
            out[name] = value
    return out


def try_observation_counterexample(env, original_scores, original_choices, rng):
    """Construct current-state alternatives; these are not claimed observed rollouts."""
    original_obs = env.observe()[0][:, 0].clone()
    old_age = env.step_index - env.tau
    for trial in range(64):
        alt = copy.copy(env)
        alt.tau = env.tau.clone()
        for e in range(env.count):
            for u in range(env.U):
                indices = torch.where(env.q[e, u])[0]
                if len(indices) > 1:
                    order = torch.as_tensor(rng.permutation(len(indices)))
                    replacement = env.tau[e, u, indices[order]]
                    cache_age = env.step_index - replacement
                    # Each pending block is newer than the last delivered update.
                    if torch.all(cache_age < env.aoi[e, u, indices]) and torch.all(replacement >= 0):
                        alt.tau[e, u, indices] = replacement
        if torch.equal(alt.tau, env.tau):
            continue
        np.testing.assert_array_equal(h.arr(alt.observe()[0][:, 0]), h.arr(original_obs))
        scores, choices = evaluate_candidates(alt)
        changed = choices != original_choices
        if bool(changed.any()):
            e = int(torch.where(changed)[0][0])
            return dict(kind='constructed_current_state_pair', episode_batch_index=e,
                        step=env.step_index, trial=trial, identical_sut_observation=True,
                        changed_variable='per-DS pending-block generation times; per-UAV maximum cache ages preserved',
                        original_candidate=h.RULES[int(original_choices[e])],
                        alternate_candidate=h.RULES[int(choices[e])],
                        original_scores=original_scores[:, e].tolist(), alternate_scores=scores[:, e].tolist(),
                        cache=env.q[e].tolist(), aoi=env.aoi[e].tolist(),
                        original_cache_age=old_age[e].tolist(), alternate_cache_age=(env.step_index-alt.tau[e]).tolist(),
                        limitation='Admissible timestamp reassignment at current state; no claim that both states were reached in original evaluation. UAV local observations differ.')
    return None


def main():
    torch.set_num_threads(1)
    h.verify()
    reads, checks, counterexample = set(), [], None
    rng = np.random.default_rng(20260911)
    scenarios = {'fixed_0': [[0, 0]], 'fixed_1': [[0, 1]], 'fixed_2': [[0, 2]],
                 'multi_0_1_2': [[0, 0], [200, 1], [400, 2]]}
    sample_slots = {0, 1, 50, 199, 200, 299, 399, 400, 599}
    for scene, schedule in scenarios.items():
        env, _, _, _ = h.make_env(h.config('joint_continue', 218), h.EVAL_SEEDS, schedule)
        for slot in range(600):
            if slot in sample_slots:
                before = immutable_fingerprint(env)
                normal_scores, normal_choices = evaluate_candidates(env)
                view = CurrentStateView(env, reads)
                guarded_scores, guarded_choices = evaluate_candidates(view)
                guarded_action, guarded_reward = rules.myopic_actions(view)
                normal_action, normal_reward = rules.myopic_actions(env)
                np.testing.assert_array_equal(h.arr(guarded_scores), h.arr(normal_scores))
                np.testing.assert_array_equal(h.arr(guarded_choices), h.arr(normal_choices))
                np.testing.assert_array_equal(h.arr(guarded_reward), h.arr(normal_reward))
                for a, b in zip(guarded_action, normal_action):
                    np.testing.assert_array_equal(h.arr(a), h.arr(b))
                assert before == immutable_fingerprint(env), 'Counterfactual scoring mutated state'
                saved = {}
                for name in ('noise_us', 'noise_sat', 'noise_du', 'potential_content', 'instructions'):
                    tape = getattr(env, name)
                    saved[name] = tape[slot+1:].clone()
                    if name == 'instructions':
                        tape[slot+1:] = (tape[slot+1:] + 1) % 3
                    else:
                        tape[slot+1:] = 123456.789
                changed_scores, changed_choices = evaluate_candidates(env)
                np.testing.assert_array_equal(h.arr(changed_scores), h.arr(normal_scores))
                np.testing.assert_array_equal(h.arr(changed_choices), h.arr(normal_choices))
                for name, saved_tail in saved.items():
                    getattr(env, name)[slot+1:] = saved_tail
                assert before == immutable_fingerprint(env)
                checks.append(dict(scenario=scene, slot=slot, states=env.count,
                                   guard='PASS', future_tape_perturbation='NO_CHANGE',
                                   all_nine_counterfactuals='IDENTICAL', no_mutation=True))
                if counterexample is None and slot >= 50:
                    counterexample = try_observation_counterexample(env, normal_scores, normal_choices, rng)
                    if counterexample:
                        counterexample['scenario_origin'] = scene
            # Use diverse reachable states under the calibrated instruction rule.
            gid = int(env.context()[0][0])
            h.checked_step(env, h.rule_actions(env, ['m5_equal', 'm0_urgency', 'm5_equal'][gid]))
        print('INFORMATION AUDIT PASS', scene, flush=True)
    h.verify()
    files = [ROOT / 'rule_tools.py', ROOT / 'helpers.py', ROOT / 'source/tensor_env.py',
             ROOT / 'source/selector_train.py', ROOT / 'source/reference/runtime/harl/envs/uav_escs/reward_objective.py',
             ROOT / 'source/episode_source.py', OUT / 'audit_runtime.py']
    attrs = {}
    for file in files[:3]:
        tree = ast.parse(file.read_text())
        attrs[str(file)] = sorted({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                                 and isinstance(n.value, ast.Name) and n.value.id == 'env'})
    result = dict(state='PASS', sampled_states=sum(x['states'] for x in checks),
                  tested_candidate_state_pairs=9*sum(x['states'] for x in checks),
                  checks=checks, dynamic_attribute_reads=sorted(reads), static_env_attributes=attrs,
                  same_sut_observation_counterexample=counterexample,
                  source_hashes={str(file): h.digest(file) for file in files},
                  conclusion='No future information read by current myopic implementation. It uses global current per-DS state and exact one-slot transition/reward model, beyond each deployed actor observation.')
    h.write(OUT / 'runtime_audit.json', result)
    print('RESULT', result['sampled_states'], 'states;', result['tested_candidate_state_pairs'], 'candidate-state pairs', flush=True)
    print('READS', sorted(reads), flush=True)
    print('OBSERVATION_COUNTEREXAMPLE', counterexample, flush=True)


if __name__ == '__main__':
    main()
