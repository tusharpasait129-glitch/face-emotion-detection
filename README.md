# Dataset

This project trains on **FER2013** (Facial Expression Recognition 2013),
35,887 grayscale 48x48 face images labelled with 7 emotions:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

The CSV file (~300MB) is not included in this repository. To obtain it:

1. Create a free account at [Kaggle](https://www.kaggle.com).
2. Download the dataset from either:
   - https://www.kaggle.com/datasets/msambare/fer2013 (image folders), or
   - https://www.kaggle.com/datasets/deadskull7/fer2013 (single CSV — this is the format the code expects)
3. Place the file here as:
   ```
   data/fer2013.csv
   ```
4. The CSV must have three columns: `emotion`, `pixels`, `Usage`
   (`Usage` values: `Training`, `PublicTest`, `PrivateTest`).

If you only have the image-folder version, convert it to this CSV format
first, or adapt `src/dataset.py` to read directly from the folder structure
(`ImageFolder`-style) — the model and training loop do not need to change.
