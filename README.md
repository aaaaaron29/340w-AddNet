# ADD-Net: Alzheimer's Disease Stage Classification

A CNN-based classifier for Alzheimer's disease stage from MRI scans, built as a two-person
course project (340W). It starts from the published ADDNet model (see Background below) and
tries to improve on it. Trained on a 6,400-image MRI dataset with SMOTETomek applied to
correct class imbalance, my version reaches 93% overall accuracy and 98% on MCI (mild
cognitive impairment) detection.

**Team:** 2 people. I was the sole contributor on data processing, model implementation,
training, and evaluation (everything in this repo). My teammate handled the written
report/presentation deliverable, which isn't included here.

## Background

ADDNet is not my own design. It comes from the paper
[*ADDNet - An Enhanced Convolutional Neural Network for Detection and Classification of
Alzheimer's Disease*](https://jowua.com/wp-content/uploads/2024/12/2024.I4.008.pdf) by
Radhakrishna Chamakuri and Dr. Hyma Janapana (Journal of Wireless Mobile Networks,
Ubiquitous Computing, and Dependable Applications, vol. 15, no. 4, December 2024,
pp. 109-132, DOI 10.58346/JOWUA.2024.I4.008). The paper proposes a CNN that improves on a
baseline CNN for four-stage Alzheimer's classification on a 6,400-image, 128 x 128 MRI
dataset. It uses data augmentation (rotation, flipping, zooming, shearing, center shifting)
and dropout, splits the data 80/20 into training and test sets, and reports 98.83% overall
accuracy with 97.23% precision, 98.54% recall, and a 97.88% F1-score.

I started from that model and worked on improving it. On top of the paper's setup, I added:

- SMOTETomek rebalancing of the imbalanced classes
- Grad-CAM heatmaps and t-SNE plots to inspect what the model is keying on, which the paper
  doesn't include

My version's 93% overall accuracy is lower than the paper's 98.83%. The two numbers aren't a
clean comparison, because my split, preprocessing, and rebalancing differ from the paper's,
and I'm not claiming to beat the paper's figure. The 98% MCI number is a per-class result
from my own evaluation.

## Dataset

[Alzheimer's Dataset (Kaggle)](https://www.kaggle.com/datasets/shahidzikria/alz-dataset/):
6,400 preprocessed MRI images across 4 classes (Non Demented, Very Mild, Mild, Moderate).

## Method

- Class imbalance corrected with SMOTETomek (combined over/under-sampling)
- CNN trained with Keras/TensorFlow
- Evaluation: accuracy, per-class metrics, Grad-CAM heatmaps and t-SNE plots to inspect
  class separation and sanity-check what the model is keying on

## A note on the train/test split

The split in this repo (`train_test_split`, 80/20/20 train/val/test) is done at the image
level, not the patient level, and this Kaggle dataset is known to contain augmented /
near-duplicate images derived from the same source scans. That means the reported accuracy
is best read as an upper bound rather than a guaranteed generalization number. A
patient-level split would be the more rigorous way to validate this. Flagging this here
instead of letting the headline number stand unqualified.

## Repo structure

This was built incrementally over a semester, so the notebooks reflect that; all intermediary notebooks are not published:

- `ADDNet126run.ipynb`: primary implementation notebook with full outputs
- `Midterm/`: midterm checkpoint submission
- `Week 12 - Modified Implementation/`: added Grad-CAM interpretability
- `FinalSubmission/`: final submitted notebook (`Group34_FinalSubmission.ipynb`)

## Stack

Python, TensorFlow, Keras, scikit-learn, imbalanced-learn (SMOTETomek), Matplotlib
