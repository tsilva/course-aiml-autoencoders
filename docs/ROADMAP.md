# Learning Roadmap

The core path is optimized for learning per minute. The runnable sequence and
short-run defaults live in [the course guide](../course/README.md).

## Core: one revealing contrast at a time

1. Inspect an untrained encoder/bottleneck/decoder on synthetic shapes.
2. Understand pixel error and compare input-ignoring reconstruction baselines.
3. Train an eight-dimensional linear AE, then change its hidden-layer setup.
4. Change training corruption and evaluate noisy-input recovery against clean targets.
5. Distinguish smooth AE interpolation from a learned sampling distribution.
6. Inspect stochastic clouds, then compare beta-zero and beta-one VAEs.
7. Increase beta and test latent dependence with a deliberately fixed-note control.
8. Inspect quantization, global symbol usage, and a learned token prior.
9. Choose a mechanism for a task and diagnose an unfamiliar result.

The default notebooks reuse exact matching short CPU runs. Tiny visible probes
come before training. Mathematical derivations, larger sweeps, reports, and
multi-seed confirmation are optional.

## Optional investigations

Pick a question rather than completing this entire list:

- AE latent-size sweep, decoder directions, PCA geometry, and identity copying.
- Sparse activity and its scale ambiguity; additional corruption types.
- Full beta sweep, latent traversals, warm-up, and free bits.
- Codebook-size and commitment sweeps; EMA and dead-code recovery are future work.
- Frozen-latent classification or retrieval against raw pixels and PCA.
- An important two-variant comparison over multiple seeds and final test confirmation.

Convolutional AE variants, EMA codebook updates, larger token priors, and
additional downstream probes are extension ideas, not implemented prerequisites.
