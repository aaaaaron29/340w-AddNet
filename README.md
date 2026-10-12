# ADD-Net: Alzheimer's Disease Stage Classification

A CNN-based classifier for Alzheimer's disease stage from MRI scans, built as a two-person
course project (340W). Trained on a 6,400-image MRI dataset with SMOTETomek applied to
correct class imbalance, reaching 93% overall accuracy and 98% on MCI (mild cognitive
impairment) detection.

**Team:** 2 people. I was the sole contributor on data processing, model implementation,
training, and evaluation (everything in this repo). My teammate handled the written
report/presentation deliverable, which isn't included here.

## Dataset

[Alzheimer's Dataset (Kaggle)](https://www.kaggle.com/datasets/shahidzikria/alz-dataset/) —
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
is best read as an upper bound rather than a guaranteed generalization number — a
patient-level split would be the more rigorous way to validate this. Flagging this here
instead of letting the headline number stand unqualified.

## Repo structure

This was built incrementally over a semester, so the notebooks reflect that:

- `ADDNet126run.ipynb` — primary implementation notebook with full outputs
- `Midterm/` — midterm checkpoint submission
- `Week 12 - Modified Implementation/` — added Grad-CAM interpretability
- `FinalSubmission/` — final submitted notebook (`Group34_FinalSubmission.ipynb`)

## Stack

Python, TensorFlow, Keras, scikit-learn, imbalanced-learn (SMOTETomek), Matplotlib
