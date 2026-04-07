# FACTS Multimodal Decomposition Dataset (FMDD)

## Dataset Access: [Dataset](https://figshare.com/s/4cda1d1bfd91ca9037ff)

FACTS Multimodal Decomposition Dataset (FMDD) is an AI‑ready, curated benchmark for postmortem interval (PMI) estimation, built from 2011–2023 donor placements at the Texas State University Forensic Anthropology Center (FACTS). It combines outdoor RGB decomposition imagery, donor metadata, and harmonized weather records to support multimodal, uncertainty‑aware PMI modeling under real‑world conditions.

## Overview

FMDD addresses the challenge of estimating time since death in outdoor environments where biological and environmental processes interact in complex ways. The dataset standardizes visual, demographic, and environmental data streams to enable reproducible research in forensic taphonomy, multimedia analysis, and biological time‑series modeling.

Key properties:

- 741 RGB images from 99 donors, covering 1-384 days post‑placement.  
- Semi‑automatic curation with foundation‑model‑assisted body segmentation and expert‑reviewed masks.  
- Image‑to‑text captions produced by a multimodal model and manually standardized for visual grounding.  
- Harmonized, gap‑filled weather records aligned to each donor’s decomposition timeline, sourced from NOAA, Open‑Meteo, and local Freeman Ranch stations.

## Dataset Contents

FMDD consists of several coordinated components:

- **RGB decomposition images**  
  High‑resolution outdoor images documenting human decomposition at the FACTS facility across multiple stages (fresh, bloat, active decay, advanced decay, skeletonization).

- **Segmentation masks**  
  Binary full‑body masks (body vs background), curated with foundation models and expert review to ensure consistent localization of remains.

- **Image captions**  
  Standardized, visually grounded textual descriptions per image, generated via a prompt‑guided vision‑language model and corrected by human annotators.

- **Donor bioprofiles**  
  De‑identified demographic and case information (e.g., age ranges, sex, basic health and contextual attributes) encoded at resolutions suitable for research and privacy.

- **Placement metadata**  
  Structured records of placement timing, experimental design, and relevant contextual information for each donor.

- **Weather and environmental histories**  
  Time‑aligned environmental covariates (such as temperature, humidity, rainfall) aggregated and gap‑filled from NOAA, Open‑Meteo, and local station data for the full decomposition window.

## Intended Uses

FMDD is intended for:

- Development and evaluation of PMI estimation models using visual, tabular, and multimodal inputs.  
- Research on decomposition dynamics and the impact of intrinsic (donor) and extrinsic (environmental) factors.  
- Benchmarking segmentation, captioning, and multimodal representation learning methods in a high‑stakes forensic setting.  
- Methodological work on uncertainty‑aware and interpretable models for biological time‑series and environmental multimedia data.

## Access and Ethical Considerations

FMDD is based on sensitive imagery of human remains and is governed by FACTS policies, institutional review processes, and donor consent agreements. Access is controlled to balance scientific value with privacy, dignity, and ethical obligations.

Main principles:

- **Controlled access**  
  - Public materials: high‑level statistics, documentation, and example code can be shared broadly.  
  - Restricted materials: full‑resolution images and detailed metadata require a vetted Data Use Agreement (DUA).

- **Permitted uses**  
  - Legitimate scientific research related to PMI estimation, decomposition modeling, forensic and environmental studies, and closely related methodological work.  
  - Redistribution of the original data and any re‑identification attempts are prohibited.

- **De‑identification and safeguards**  
  - Direct identifiers are removed; demographic and temporal attributes are coarsened to minimize re‑identification risk.  
  - Users are encouraged to employ secure compute environments and to avoid generative uses that could yield realistic depictions of specific individuals.

Refer to the official DUA and dataset documentation for complete terms and conditions.


## Directory Structure

This repository is intended for dataset documentation, metadata schemas, and helper utilities rather than model training code.

```text
fmdd-dataset/
├─ docs/               # Dataset description, DUA templates, ethical guidelines
├─ metadata/           # Public metadata schemas, example records
├─ examples/           # Example scripts for loading and inspecting metadata (no raw images)
└─ README.md
```

## Citation

If you use FMDD in your work, please cite the associated ACM Multimedia paper:

> FACTS Multimodal Decomposition Dataset (FMDD): AI‑Ready RGB, Metadata, and Weather Benchmark for Postmortem Interval Estimation. Proceedings of ACM Multimedia (ACM MM ’26), Rio de Janeiro, Brazil, 2026.

**(Replace this with the final ACM citation and DOI once available.)**
