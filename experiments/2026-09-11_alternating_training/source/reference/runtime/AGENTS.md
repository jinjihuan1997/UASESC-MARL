# AGENTS.md

- Do not preserve backward compatibility for obsolete code paths, APIs, or configurations. Remove legacy implementations instead of adding compatibility layers or fallbacks. Preserve reproducibility through versioned code, configurations, checkpoints, and experiment records.

- Choose the simplest implementation that fully meets the current requirements. Avoid unnecessary abstractions, configuration, and indirection.

- Keep components modular and responsibilities clearly separated.

- Prefer existing project dependencies and established, well-maintained libraries over custom implementations. Do not reimplement common functionality without a clear reason.

- Preserve experimental reproducibility. Do not silently change datasets, data splits, random seeds, evaluation metrics, baselines, or evaluation protocols.

- Make only the changes necessary for the current task. Do not modify unrelated modules or experimental settings.

- Do not delete or overwrite datasets, checkpoints, experiment results, or existing configurations unless explicitly requested.

- Keep the current codebase clean and internally consistent. Remove obsolete or duplicated implementations when they are no longer needed.

- After changes, run the relevant lightweight tests or smoke tests. Clearly report what was changed, which files were modified, and the exact commands needed to reproduce or validate the result.
