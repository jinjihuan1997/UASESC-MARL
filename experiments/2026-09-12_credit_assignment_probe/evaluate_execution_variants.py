from support import *
from engine import run_batch


def main():
    m = verify_inputs(); pre = read(HERE / 'preflight.json')
    assert pre['state'] == 'PASS'
    count = 0
    for seed in m['training_seeds']:
        for scenario in m['scenarios']:
            for variant in 'ABCDEF':
                for action_seed in m['action_seeds'] if variant in 'BCD' else [0]:
                    folder = HERE / 'execution' / f'seed_{seed}' / scenario / f'{variant}_action{action_seed}'
                    result = run_batch(folder, seed, scenario, variant, action_seed, m['validation_seeds'], pre['execution_device'])
                    count += result['episodes']
                    save_status('execution', completed_execution_episodes=count, total_execution_episodes=2280)
    for name in ('R_instruction', 'R_equal_instruction'):
        for scenario in m['scenarios']:
            result = run_batch(HERE / 'execution/rules' / name / scenario, m['training_seeds'][0], scenario,
                               name, 0, m['validation_seeds'], pre['execution_device'])
            count += result['episodes']
            save_status('execution', completed_execution_episodes=count, total_execution_episodes=2280)
    assert count == 2280
    verify_inputs()
    write(HERE / 'execution/complete.json', dict(state='complete', episodes=count, physical_steps=count*600, updated_utc=stamp()))


if __name__ == '__main__': main()
