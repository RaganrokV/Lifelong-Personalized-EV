# Pretrain-Enhanced Lifelong Learning Sustaining Personalized Energy Prediction Across Electric Vehicle Lifecycles
#### This repository provides the official implementation of our study，which introduces a **pretrain-enhanced lifelong learning framework** for personalized and adaptive energy prediction in electric vehicles (EVs).  
#### Our approach integrates:
- **Large-scale self-supervised pretraining** on 1,633 EVs and 5.6 million trips.  
- **Incremental lifelong adaptation** for dynamic personalization across driver–vehicle–environment contexts.  
- **Unified evaluation metrics** covering *stability*, *plasticity*, and *uncertainty* to standardize lifelong learning assessment.

This work bridges **AI, electrochemistry, and transportation systems engineering**, offering a scalable route toward self-evolving, sustainable AI systems for mobility.

## Overview
<img width="994" height="856" alt="image" src="https://github.com/user-attachments/assets/1a19deca-dab5-4f8f-8d77-8df2466bb952" />

##  Key Features
-  **Pretrain-Enhanced Encoder** — Generalizable probabilistic representations distilled from millions of driving samples.  
-  **Incremental Learner** — Supports continual adaptation to evolving driver behavior and battery conditions.  
-  **Unified Benchmark** — Includes standardized protocols for evaluating lifelong learning in regression-based prediction tasks.  
-  **Efficient Inference** — Edge-deployable (≤0.03 s/trip with Monte Carlo dropout, n=50).  

# How to start?

#### 1. please download our pre-trained model on [OneDrive](https://1drv.ms/u/c/284956e407934917/EW_79LiVimRHvlc6Ne1Zi1EBV_90rNBWObv05X33l7ZJTw)
#### 2. Then, loading a pre-trained model with the code：40-47 （choose what you like）
#### 3. Start your own lifelong learning！
