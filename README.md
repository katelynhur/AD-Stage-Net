# AD-Stage-Net: Hybrid Deep Learning for Four-Stage Alzheimer’s MRI Classification  
**Katelyn Hur, Red River High School, Grand Forks, North Dakota**

## Abstract
<i>Alzheimer’s disease (AD) MRI classification often suffers from limited model generalization. Deep learning models were developed to distinguish four stages (No AD, Very Mild, Mild, and Moderate) from MRIs. One Kaggle MRI dataset was used for training, with independent testing from two other datasets. The best ensemble model (ResNet50_DenseNet161) achieved 98.71% accuracy. Models were deployed to a web platform for MRI upload and real-time classification, demonstrating the potential of robust, transparent clinical decision support.</i> 

## Introduction
Alzheimer’s disease (AD) is the most prevalent neurodegenerative disorder and the leading cause of dementia, affecting up to 131.5 million people by 2050. AD is commonly diagnosed after displaying symptoms, so early and accurate classification of AD severity is essential for effective intervention. MRI captures structural atrophy relevant to AD staging, and CNN-based models have shown promise for automatic classification; however, reproducibility, dataset variability, and accessibility remain challenges. We address cross-dataset generalization by training on one public dataset and evaluating on two independent datasets, and we deploy models (individual and ensemble hybrids) for public use to promote transparency and utility. 

## Methods
Three publicly available datasets on Kaggle and Hugging Face were used. Training data came from the Kaggle [“Best Alzheimer’s MRI” (Luke) set](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy), and evaluation used the Luke test split and two independent sets from Kaggle ([“Alzheimer MRI 4 classes,” Marco](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)) and Hugging Face ([“Alzheimer_MRI,” Falah](https://huggingface.co/datasets/Falah/Alzheimer_MRI)). Labels were harmonized across datasets. We compared CNN_Small, ResNet-50/101/152, DenseNet-121/161/169/201, ResNeXt-50/101, EfficientNet-B0, MobileNetV2/V3, VGG16-BN, and InceptionV3. Images were padded to square (Marco), then resized to 224x224 pixels (299x299 for InceptionV3), and normalized to ImageNet statistics. Best hyperparameters identified for ResNet50 were applied across other architectures during fine-tuning. Models were trained on Luke with an 80/20 train-validation split using AdamW, label smoothing, early stopping, and saving the best-validation checkpoint; inference used strict state-dict loading. Late-fusion ensembles, averaging logits across strong singles and selected models, were explored as well to maximize accuracy across datasets. 

## Results
Strong single models generalized across datasets, with DenseNet, ResNeXt, and ResNet families performing consistently well; the smaller baseline (CNN_Small) lagged. Accuracy was highest on Luke and showed modest, dataset-related drops on Marco/Falah, reflecting dataset differences. The best ensemble (ResNet50_DenseNet161) achieved a 98.71% average accuracy and the highest minimum accuracy across datasets, indicating superior robustness. Across the Top 10 models, rankings were relatively stable from Luke to Marco/Falah, with DenseNet and ResNeXt frequently appearing among the leaders. The ensemble consistently improved worst-case performance relative to its constituent singles, supporting the strategy of maximizing the minimum accuracy across datasets. 

## Discussion and Conclusions
Across the three datasets, most single models and the ensemble achieved high accuracy, with overall differences of roughly 3%. Some singles were slightly higher on Luke and Falah, but they dropped more on Marco, while the ensemble stayed strong on all three sets, reducing the worst-case dip and suggesting better real-world robustness. Consistent preprocessing and strict checkpoint loading improved reproducibility. Limitations include structural MRI only and modest public cohorts. Next steps are to expand training and testing to larger, more diverse datasets (e.g., OASIS, ADNI), add explainability methods such as Grad-CAM, and pursue clinical validation. [Our public demonstration website](https://huggingface.co/spaces/katelynhur/AD-Stage-Net) further shows the feasibility of broader testing and community use. Overall, AD-Stage-Net is a practical, robust approach for four-stage Alzheimer’s MRI classification.


---

## Highlights
- Four-class AD staging: **No AD / Very Mild / Mild / Moderate**  
- Hybrid **ResNet50_DenseNet161** model achieved **98.71%** average accuracy on three different test sets 
- Lightweight preprocessing and reproducible training loop  
- **Live demo available**: [Hugging Face Space](https://huggingface.co/spaces/katelynhur/AD-Stage-Net)  

---

## Directory Structure

- **Code/**  
  Contains Python and shell scripts for running the models on a local machine.

- **Notebook/**  
  Jupyter Notebooks tailored for use on Google Colab, including training and evaluation workflows.

- **Results/**  
  Select summary outputs from experiments, including leaderboards and best-performing model details.

- **Data/**  
  Downloaded compressed MRI image files used in this study, obtained from Kaggle and Hugging Face.

- **HuggingFaceSpaces/**  
  Files supporting deployment of the live demo on Hugging Face Spaces.

---

## Disclosure
- OpenAI ChatGPT 5 and Google Gemini 2.5 assisted in writing the Python and Jupyter Notebook codes in this repository.
