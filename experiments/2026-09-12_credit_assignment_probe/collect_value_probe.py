from support import *
from engine import run_batch


def main():
    m = verify_inputs(); pre = read(HERE / 'preflight.json'); assert pre['state'] == 'PASS'
    count = 0
    seeds = m['probe_fit_seeds'] + m['probe_holdout_seeds']
    assert not set(seeds).intersection(m['validation_seeds'] + m['training_seeds'] + m['preflight_seeds'] + m['reserved_final_test_seeds'])
    for parent in m['training_seeds']:
        for action in m['probe_action_seeds']:
            result = run_batch(HERE / 'probe' / f'seed_{parent}' / f'action{action}', parent, 'random_switch_once',
                               'D', action, seeds, pre['execution_device'], probe=True)
            count += result['episodes']
            save_status('probe_collection', completed_execution_episodes=2280, completed_probe_episodes=count, total_probe_episodes=144)
    assert count == 144
    verify_inputs()
    write(HERE / 'probe/complete.json', dict(state='complete', episodes=count, physical_steps=86400, updated_utc=stamp()))


if __name__ == '__main__': main()
