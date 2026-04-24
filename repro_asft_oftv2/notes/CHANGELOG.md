# Changelog

## Repro bundle creation

This folder was created to package the already working experiment into a smaller reproducible layout.

Changes made for reproducibility only:

- copied the minimal training sources into `src/`
- copied the tiny adapter smoke test into `tests/`
- added `run/*.sh` wrappers for smoke, full training, eval, and merge guidance
- added `configs/*.env` templates
- added this documentation set

No new training algorithm, adapter behavior, loss behavior, or evaluation logic was added here.
