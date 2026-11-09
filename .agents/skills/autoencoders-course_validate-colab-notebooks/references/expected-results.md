# Expected results by lesson

Use these as review anchors, then read the current canonical lesson: its learning
objective and predictions are authoritative if the curriculum changes. Record
observed values rather than hard-coding approximate training results. Compare
reconstruction MSE only with MSE, and summed BCE only with summed BCE.

| ID | Numerical/structural evidence | Visual/learning evidence |
| --- | --- | --- |
| 00 | Input and reconstruction shapes match; batch size is two and latent shape agrees with the smoke recipe. Outputs are finite. | The image → smaller note → image pipeline renders. This is an untrained model; poor reconstruction is expected. |
| 01 | Toy errors are approximately `[0, 0.04, 0.24, 0.12]`. The training mean-image baseline beats all-black MSE on held-out data. | Mean reconstructions are the same poster for every input. Shifted pixels cost more than one missing pixel, even when the shape looks similar. |
| 02 | Latent width is eight. Held-out linear-AE MSE is below the mean-image baseline; record both and the fraction of error removed. Learning curves are finite and improve. | Reconstructions respond to different inputs and recover coarse structure, with lost detail visible in error maps. |
| 03 | Linear and nonlinear runs use the same data protocol and latent width. Record both held-out MSEs and parameter counts. | Compare reconstructions; explain the observed capacity change and parameter-count confound. A reversed short-run ranking needs investigation, not fabricated improvement. |
| 04 | Clean and denoising controls differ only in declared input corruption. Both use identical seeded evaluation noise and clean targets. Record clean MSE, corrupted-input MSE, and unprocessed noisy-input MSE. Check whether denoising improves the noisy input and clean-trained control. | The displayed triplets show noisy input → prediction → clean target. Inspect actual noise removal and remaining blur. Flag an absent teaching contrast. |
| 05 | The nine-step interpolation has the original encoded endpoints; decoded and random-code images have valid, finite shapes. | Compare walks between encoded examples with arbitrary standard-normal codes. Explain why reconstruction training alone does not specify a sampling distribution; random outputs need not always look worse. |
| 06 | Toy samples center near `(2, -1)`; spread 0.8 is visibly/numerically wider than spread 0.1. The trained objective has beta zero and finite reconstruction/raw-KL metrics. | Distinguish the illustrative clouds from trained latent behavior. Reconstruction can work without establishing useful standard-normal sampling. |
| 07 | Compare beta-zero and beta-one runs with the same split and architecture. Record raw KL and reconstruction BCE separately; beta one should exert pressure toward the shared prior. | Inspect both random-latent grids. Do not claim that lower KL alone guarantees good generated images. Investigate contradictory or absent contrasts. |
| 08 | Exactly two observations, beta 1 and 4. Rate is raw KL; distortion is center-decoded BCE. Record active dimensions and actual rate/distortion changes; confirm only beta changed. | Plot labels and annotations match the measured points. Greater KL pressure usually lowers rate at a reconstruction cost; short runs are not an optimal frontier. |
| 09 | Normal and fixed-note reconstructions have equal valid shapes. Record their MSEs and per-dimension KL. With a zero note, decoded outputs should be identical across examples. | Compare normal versus fixed-note image grids. Use the measured effect as evidence of decoder reliance; do not label low KL alone as posterior collapse. |
| 10 | The fixed-codebook toy chooses symbols `[0, 0, 1, 2]`; snapped points match those embeddings. Trained token indices lie within the codebook range and usage metrics are finite. | Reconstruction grids, spatial token maps, and uniform-token samples render. Explain that arbitrary valid tokens need not form a plausible image. |
| 11 | Both toys use all eight codes. Balanced perplexity is approximately 8; dominated perplexity is near 1 and much smaller. Actual usage has `1 <= perplexity <= codes used <= codebook size` within numerical tolerance. | Usage histogram agrees with summary counts. Distinguish occupancy from balanced usage and useful representation. |
| 12 | Record uniform, training-frequency unigram, and learned-prior held-out cross-entropies. The prior is paired with the exact VQ checkpoint. A lower learned CE than unigram demonstrates arrangement information; beating only uniform is insufficient. | Compare uniform-token samples with learned-prior samples and inspect the learning curve. If it does not beat unigram, report that arrangement learning is unproven at this budget. |
| 13 | Three labeled, correctly matched AE/VAE/VQ-VAE run paths resolve to their intended configurations and figures. | Compare strengths/limitations for the task without ranking incompatible objectives or treating reconstruction as proof of generation. |

For a missing contrast, first check the configuration, split, checkpoint, metric
definition, and training trajectory. Preserve the observed result and report
whether it is a runtime defect, weak educational evidence, or an explained
limitation. Changes to training budgets or lesson claims require a new snapshot
and rerun; the quick learner path must be evaluated at its actual default budget.
