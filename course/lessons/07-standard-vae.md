# Lesson 07 — Give the clouds a shared address system

**Learning objective:** explain why prior pressure trades reconstruction detail for easier sampling.

Imagine every input's cloud using a different, far-away address system.
Random notes drawn near zero will miss many of them. KL pressure charges for
moving or narrowing a cloud away from the shared standard-normal prior.
That encourages overlap with places our sampler knows how to visit.

[Open the notebook](../notebooks/07-standard-vae.ipynb). Run its cells in order for the experiment and evidence.

## Predict before running

Compared with beta zero, should prior pressure improve every reconstruction and every sample?

<details>
<summary>Reveal the expected reasoning</summary>

No. It can improve prior compatibility while sacrificing reconstruction information. Too much pressure can make clouds lose their input-specific information.
</details>

## Run the revealing experiment

The notebook uses short CPU runs and reuses exact matching completed evidence.
It prints the run path and measured duration. Keep `PROFILE = "quick"` for the
core path; use `"full"` for the original budget. No worksheet is required.

## Advancement gate — transfer check

All input clouds match the prior perfectly, and every reconstruction looks alike. Is zero KL a success?

<details>
<summary>Reveal the expected reasoning</summary>

It indicates input information was lost: the decoder is not reconstructing distinct inputs. Check latent dependence and reconstruction evidence, not KL alone.
</details>

<details>
<summary>Optional: go deeper</summary>

The beta-one negative ELBO combines expected negative log likelihood with
$D_{KL}(q(z|x)\|p(z))$. Beta changes their relative price.
Average posterior-to-prior KL includes input information **and** aggregate
posterior mismatch; it is not an exact measurement of mutual information.
Nonzero per-dimension KL is a diagnostic, not proof a coordinate carries useful
input information. See the [objective reference](../lessons/07-standard-vae.md).
</details>

**Next:** [Lesson 08](../notebooks/08-rate-distortion.ipynb).
