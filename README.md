# AD-Stage-Net: Hybrid Deep Learning for Four-Stage Alzheimer’s MRI Classification  
**Katelyn Hur, Red River High School, Grand Forks, North Dakota**

## Abstract
**Background.** Early and accurate staging of Alzheimer’s disease (AD) from MRI is essential for timely intervention, but model generalization is often limited by dataset variability and small sample diversity.  

**Objective.** To develop and evaluate hybrid deep learning models that classify four AD stages (**No AD, Very Mild, Mild, Moderate**) from structural MRIs, and to deploy the best models for real-time clinical decision support.  

**Methods.** Two publicly available Kaggle MRI datasets were combined to increase training diversity. Preprocessing included resizing, normalization, and light augmentations (horizontal flips, ±10° rotations). Multiple convolutional neural network (CNN) families (*ResNet, DenseNet, EfficientNet, Inception*) were trained, and **hybrid models** were created by fusing complementary backbones. Models were evaluated on a held-out validation set and an **independent test set**.  

**Results.** Several models surpassed **99.9% accuracy** on the independent test set, with the best hybrid (**ResNet50_InceptionV3**) achieving **100%** accuracy. Confusion matrices demonstrated near-perfect class separation. Training stability was enhanced by early stopping and learning-rate scheduling. A live demo was deployed as a Hugging Face Space: [https://huggingface.co/spaces/katelynhur/AD-Stage-Net](https://huggingface.co/spaces/katelynhur/AD-Stage-Net).  

**Conclusions.** Hybrid CNN models offer state-of-the-art performance for four-stage AD MRI classification and can be deployed in real time. Further testing on patient-level, multi-site clinical data is needed to confirm generalizability.  

---

## Highlights
- Four-class AD staging: **No AD / Very Mild / Mild / Moderate**  
- Hybrid **ResNet50_InceptionV3** model achieved **100%** independent test accuracy  
- Multiple models ≥ **99.9% accuracy**  
- Lightweight preprocessing and reproducible training loop  
- **Live demo available**: [Hugging Face Space](https://huggingface.co/spaces/katelynhur/AD-Stage-Net)  

---

## Availability
- **Demo:** [https://huggingface.co/spaces/katelynhur/AD-Stage-Net](https://huggingface.co/spaces/katelynhur/AD-Stage-Net)  
- **Code & Models:** This GitHub repository  
