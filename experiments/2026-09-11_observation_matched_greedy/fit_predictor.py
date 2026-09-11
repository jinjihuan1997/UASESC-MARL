"""Fit immediate-score estimates on calibration seeds only; export numeric trees."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    folder = Path(args.branch)
    protocol = json.loads((folder.parent / 'protocol.json').read_text())
    with np.load(folder / 'calibration_data.npz') as z:
        X, y, seeds, gid = [z[k] for k in ('X', 'y', 'seeds', 'gid')]
    train = np.isin(seeds, protocol['fit_seeds'])
    validation = np.isin(seeds, protocol['validation_seeds'])
    assert (train ^ validation).all()
    assert not np.isin(seeds, protocol['evaluation_seeds']).any()
    reg = HistGradientBoostingRegressor(**protocol['predictor_parameters'])
    reg.fit(X[train], y[train])
    predicted = reg.predict(X[validation])
    labels = y[validation]
    mse = float(np.mean((predicted-labels)**2))
    nodes = [predictors[0].nodes for predictors in reg._predictors]
    assert all(not node['is_categorical'].any() for node in nodes)
    width = max(map(len, nodes))
    payload = {}
    mapping = dict(feature='feature_idx', threshold='num_threshold', left='left', right='right',
                   leaf='is_leaf', value='value')
    for new, old in mapping.items():
        array = np.zeros((len(nodes), width), dtype=nodes[0][old].dtype)
        for i, node in enumerate(nodes):
            array[i, :len(node)] = node[old]
        payload[new] = array
    payload['baseline'] = np.asarray(float(reg._baseline_prediction[0, 0]))
    payload['depth'] = np.asarray(max(int(n['depth'].max()) for n in nodes))
    indices = np.flatnonzero(validation)[::max(1, int(validation.sum())//1000)]
    check_X, native = X[indices], reg.predict(X[indices])
    position = np.zeros((len(check_X), len(nodes)), dtype=int)
    trees = np.arange(len(nodes))[None]
    for _ in range(int(payload['depth'])):
        left = check_X[np.arange(len(check_X))[:, None], payload['feature'][trees, position]] <= payload['threshold'][trees, position]
        next_pos = np.where(left, payload['left'][trees, position], payload['right'][trees, position])
        position = np.where(payload['leaf'][trees, position], position, next_pos)
    assert payload['leaf'][trees, position].all()
    portable = payload['value'][trees, position].sum(-1) + payload['baseline']
    np.testing.assert_allclose(native, portable, atol=1e-10, rtol=0)
    np.savez_compressed(folder / 'predictor.npz', **payload)
    np.savez_compressed(folder / 'predictor_validation.npz', X=check_X, native_prediction=native)
    candidate_id = X[:, -3:].argmax(-1)
    means = [[float(y[train & (gid == g) & (candidate_id == a)].mean()) for a in range(3)] for g in range(3)]
    best = np.asarray(means).argmax(-1).tolist()
    statistics = dict(state='complete', fit_rows=int(train.sum()), validation_rows=int(validation.sum()),
                      validation_rmse_score_x100=mse**.5, validation_mean_error_score_x100=float((predicted-labels).mean()),
                      trees=len(nodes), maximum_depth=int(payload['depth']), numeric_export_verified=True,
                      resource_family_means_by_instruction=means, static_family_by_instruction=best,
                      predictor_sha256=hashlib.sha256((folder/'predictor.npz').read_bytes()).hexdigest())
    (folder / 'fit.json').write_text(json.dumps(statistics, indent=2)+'\n')
    print('FIT COMPLETE', folder.name, statistics, flush=True)


if __name__ == '__main__':
    main()
