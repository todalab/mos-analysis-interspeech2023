# Analysis of mean opinion scores in subjective evaluation of synthetic speech based on tail probabilities

This repository contains source codes for the following paper.

"Analysis of mean opinion scores in subjective evaluation of synthetic speech based on tail probabilities".  
Yusuke Yasuda and Tomoki Toda  
Interspeech 2023


- [toy_data.ipynb](toy_data.ipynb): Experiments about toy data.
- [voicemos.ipynb](voicemos.ipynb): Experiments about Voice MOS challenge 2022.

# Preprocess of Voice MOS challenge 2022

The following command generates `voicemos_confint.csv` and `n_insignificant.csv`.
You need to download files of Voice MOS challenge to run it.

```
python main_confidence.py -i voicemos2022/main/DATA/sets/TRAINSET -o voicemos_confint.csv -p  n_insignificant.csv
```

