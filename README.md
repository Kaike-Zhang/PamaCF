# PamaCF
This paper has been accepted by NeurIPS 2024. [Link to the paper on Arxiv](https://arxiv.org/pdf/2410.22844)

## Authors
- **Kaike Zhang**
- Qi Cao
- Yunfan Wu
- Fei Sun
- Huawei Shen
- Xueqi Cheng

## Abstract
Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.

## Environment
- python >= 3.8
- numpy >= 1.22.2
- scikit-learn >= 1.0.2
- scipy >= 1.8.0
- torch >= 1.10.1


## Usage (Quick Start)
1. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the main script with the desired backbone model and dataset:

    ```bash
    python main.py --model=<backbone model> --dataset=<dataset>
    ```

   Replace `<backbone model>` with the name of your model, and `<dataset>` with the name of your dataset.
