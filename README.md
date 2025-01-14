# Handling Imbalanced Datasets

Imbalanced datasets are common in real-world scenarios, where the distribution of classes is skewed, making it challenging to train machine learning models effectively. Below are some techniques for handling imbalanced datasets, including **SMOTE**, **ADASYN**, **Borderline-SMOTE**, and **Tomek Links**.

---

## 1. SMOTE (Synthetic Minority Oversampling Technique)

### Overview
SMOTE is a popular oversampling technique that generates synthetic samples for the minority class. Instead of duplicating minority class samples, it creates new samples by interpolating between existing ones.

### How It Works:
1. For each sample in the minority class, identify its k-nearest neighbors.
2. Randomly select one or more neighbors.
3. Generate a synthetic sample by interpolating between the original sample and the selected neighbor.

### Key Points:
- SMOTE reduces overfitting caused by simply duplicating minority samples.
- It is effective for numerical datasets but may not work well with categorical features.

### Implementation (Python Example):
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
```

---

## 2. ADASYN (Adaptive Synthetic Sampling)

### Overview
ADASYN is an extension of SMOTE that focuses on generating synthetic samples for minority class samples that are harder to classify. It adapts the sampling process based on the distribution of the data.

### How It Works:
1. Compute the class imbalance ratio.
2. Identify minority class samples that are harder to classify (i.e., samples with fewer neighbors of the same class).
3. Generate more synthetic samples for these harder-to-classify instances.

### Key Points:
- ADASYN shifts the data distribution closer to balance by focusing on difficult samples.
- It helps improve classifier performance on minority classes.

### Implementation (Python Example):
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN()
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

---

## 3. Borderline-SMOTE

### Overview
Borderline-SMOTE is a variation of SMOTE that focuses on generating synthetic samples near the decision boundary, where minority class samples are most vulnerable to misclassification.

### How It Works:
1. Identify minority class samples near the decision boundary by checking their nearest neighbors.
2. Generate synthetic samples only for these borderline samples.

### Key Points:
- Borderline-SMOTE reduces the risk of introducing noise by avoiding oversampling in safe regions.
- It is particularly effective for improving the decision boundary between classes.

### Implementation (Python Example):
```python
from imblearn.over_sampling import BorderlineSMOTE

borderline_smote = BorderlineSMOTE()
X_resampled, y_resampled = borderline_smote.fit_resample(X, y)
```

---

## 4. Tomek Links

### Overview
Tomek Links is an undersampling technique that removes borderline instances to create a cleaner decision boundary. A Tomek Link is a pair of samples from different classes that are each other’s nearest neighbors.

### How It Works:
1. Identify all Tomek Links in the dataset.
2. Remove the majority class sample in each Tomek Link pair.

### Key Points:
- Tomek Links help reduce class overlap and improve the separability of classes.
- It is often combined with oversampling techniques like SMOTE for better results.

### Implementation (Python Example):
```python
from imblearn.under_sampling import TomekLinks

tomek_links = TomekLinks()
X_resampled, y_resampled = tomek_links.fit_resample(X, y)
