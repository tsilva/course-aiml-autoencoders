# Branding assets

The approved identity is an embroidered navy patch with cream stitching,
cyan mountains, an amber sun, and matching pixel garments connected through
cyan, violet, and amber latent-code tiles. The logo reads **AI/ML Course** on
the first line and **Autoencoders** on the second.

- `sources/logo-source.png`: transparent primary artwork.
- `sources/icon-source.png`: simplified, text-free reconstruction motif.
- `sources/social-source.png`: transparent wide embroidered plaque.
- `logo/logo-1024.png` and `icon/icon-1024.png`: square exports.
- `web-seo/`: PNG favicons, a multi-size ICO, touch/installable icons,
  a 1200 × 630 social image, and a web manifest.
- `../logo.png`: README export with a 24 px transparent margin.
- `manifest.json`: output paths and actual dimensions.

Artwork was generated with native Codex ImageGen. The prompt set is recorded
in [sources/prompts.md](sources/prompts.md). Final sources were extracted from
flat magenta backgrounds with the installed ImageGen chroma-key helper, using
`--auto-key border --tolerance 64 --edge-contract 1 --despill`. A hard matte
preserves violet interior tiles that the helper's soft matte treated as spill.

Exports use the `create-image-assets` skill's `scripts/derive_assets.py`, with
all three sources, project name `AI/ML Course: Autoencoders`, and target
`web-seo`. After export, the root logo is cropped and padded, its dimensions
are reconciled in the manifest, and the ICO is saved from the 48 px image with
embedded 16, 32, and 48 px sizes. The web manifest uses the short name
`Autoencoders`, navy theme/background colors, and `purpose: any` for icons.

The README already consumes `../logo.png`. This repository has no website
integration; favicon, web manifest, and social metadata wiring can use these
files if a site is added. Light, off-white, dark, and checkerboard previews,
128/256 px logo previews, and small favicon exports were inspected.
