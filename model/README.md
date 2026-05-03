## FMDD Dataset Generation and Benchmarked Models

This repository contains code for FMDD dataset generation and benchmarked models for image captioning, segmentation, and classification.

### Overview

This repository has two main components:

1. **Dataset Generation**
   - Automated image caption generation using curated PMI-based prompts and a Qwen2-based vision-language model
   - Automated mask generation using DINO and SAM3
   - Generated annotations are later cleaned and corrected by expert annotators

2. **Benchmarked Models**
   - BLIP-based image captioning model
   - DeepLab-based segmentation model
   - Classification model using image features and weather metadata
