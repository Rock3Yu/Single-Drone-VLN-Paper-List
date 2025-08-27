# Single-Drone-VLN-Paper-List

v0.1: 26 Aug, 25

| Paper                                                        | Year       | Task expt. VLN           | Code/Link                                                    |
| ------------------------------------------------------------ | ---------- | ------------------------ | ------------------------------------------------------------ |
| AerialVLN : Vision-and-Language Navigation for UAVs          | 2023       | Simulator                | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.pdf) [github](https://github.com/AirVLN/AirVLN) |
| Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology | 2024       | Simulator                | [website](https://prince687028.github.io/OpenUAV/)           |
| CityNav: A Large-Scale Dataset for Real-World Aerial Navigation | 2024       | Simulator (real2sim)     | [website](https://water-cookie.github.io/city-nav-proj/)     |
| GOMAA-Geo: GOal Modality Agnostic Active Geo-localization    | NeurIPS'24 |                          | [paper](https://arxiv.org/abs/2406.01917) [github](https://github.com/mvrl/GOMAA-Geo) |
| NEUSIS: A Compositional Neuro-Symbolic Framework for Autonomous Perception, Reasoning, and Planning in Complex UAV Search Missions | RA-L'24    |                          | [paper](https://arxiv.org/abs/2409.10196)                    |
| Say-REAPEx: An LLM-Modulo UAV Online Planning Framework for Search and Rescue | CoRL'24    | SAR planning scalability | [paper](https://openreview.net/forum?id=9WdUqvE03f)          |
| Exploring Spatial Representation to Enhance LLM Reasoning in Aerial Vision-Language Navigation | 2025       |                          | [paper](https://arxiv.org/abs/2410.08500)                    |
| Neuro-LIFT: A Neuromorphic, LLM-based Interactive Framework for Autonomous Drone FlighT at the Edge | 2025       | energy-efficient         | [paper](https://arxiv.org/abs/2501.19259v1)                  |
| ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions | RA-L'25    |                          | [paper](https://arxiv.org/abs/2409.10283) [github](https://github.com/souravsanyal06/ASMA) |
| OpenFly : A Versatile Toolchain and Large-scale Benchmark for Aerial Vision-Language Navigation | 25         | Simulator                | [website](https://shailab-ipec.github.io/openfly/)           |

*Only contain paper after 2022

**Other tasks belong to Navigation & Planning: [citation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10531702)

1. Language-Guided Navigation: 	navigate in envs through natural language (a goal or a specific order)
2. Vision-Language Localization: 	determine your position and posture 
3. Motion Planning: 				given position and the goal point, find a good path
4. Trajectory Prediction: 			predict the future trajectory of other agents

***Find survey papers in references



## Summary of Ref [1] 2022

VLN Method:

1. **Representation Learning**
   1. Use pretrained models
      * Vision (ResNet, Vision Transformer), Language (BERT, GPT and further pretrained)
      * Vision and Language (ViLBERT)
      *  VLN (VLN-BERT, PREVALENT, Airbert)
   2. Semantic understanding: high-level features (e.g. route structure, detected objects, specific tokens within instructions) outperform low-level features
   3. Graph representation (GNN): encode the relation between text and vision
   4. Memory-augmented model: utilize the accumulated info during navigation (LSTMs, memory model, HAMT)
2. **Action Strategy Learning**
   1. RL (RCM, reward design(CLS, nDTW, natural language), model-based, IL+ML), sparse reward signals problem.
   2. RL (Pathdreamer) exploration v.s. exploitation problem
   3. Plan future navigation steps by vision or text
3. **Data-Centric Learning** (effectively utilize the existing data, or create synthetic data)
   1. Data augmentation
      * Path-instruction pairs augmentation: better instruction quality (a speaker module, alignment scorer, adversarial discriminator, maybe out-of-fashion due to LLM?)
      * Environment augmentation: randomize, alleviate overfitting
   2. Curriculum learning: (BabyWalk, re-arrange R2R dataset) gradually increase the task's difficulty
   3. Multitask learning: benefit from cross-task knowledge
   4. Instruction Interpretation
4. **Prior Exploration** seen env -> unseen env



## Summary of Ref [2] 2024

papers mentioned in this papers are not included in the table above yet. todo



## Summary of Ref[3] 2025

todo



## Survey References

[1] Gu, J., Stefani, E., Wu, Q., Thomason, J., & Wang, X. (2022, May). Vision-and-Language Navigation: A Survey of Tasks, Methods, and Future Directions. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 7606-7623). [github](https://github.com/eric-ai-lab/awesome-vision-language-navigation) [paper](https://arxiv.org/abs/2203.12667)

[2] Zhang, Y., Ma, Z., Li, J., Qiao, Y., Wang, Z., Chai, J., ... & Kordjamshidi, P. (2024). Vision-and-Language Navigation Today and Tomorrow: A Survey in the Era of Foundation Models. *Transactions on Machine Learning Research*. [github](https://github.com/zhangyuejoslin/VLN-Survey-with-Foundation-Models) [paper](https://arxiv.org/abs/2407.07035)

[3] Tian, Y., Lin, F., Li, Y., Zhang, T., Zhang, Q., Fu, X., ... & Wang, F. Y. (2025). UAVs meet LLMs: Overviews and perspectives towards agentic low-altitude mobility. *Information Fusion*, *122*, 103158. [github](https://github.com/Hub-Tian/UAVs_Meet_LLMs) [paper](https://arxiv.org/abs/2501.02341)

