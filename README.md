# IT2011 – AI & Machine Learning — Group KU-22

## Progress Review I: Data Preprocessing and EDA

### Project overview
This project applies supervised/unsupervised ML techniques to a **lung cancer
patient dataset** (assigned by the module coordinators; sourced from
[Kaggle](https://www.kaggle.com/datasets/khwaishsaxena/lung-cancer-dataset))
to explore factors related to patient survival, as required by the IT2011
Group Assignment Specification.

### Dataset
- **File:** `data/raw/Lung Cancer.csv`
- **Rows / columns:** 890,000 rows × 17 columns
- **Target variable:** `survived` (0 = did not survive, 1 = survived; ~78% / 22% split)
- **Key features:** `age`, `gender`, `country`, `cancer_stage`, `family_history`,
  `smoking_status`, `bmi`, `cholesterol_level`, `hypertension`, `asthma`,
  `cirrhosis`, `other_cancer`, `treatment_type`, `diagnosis_date`,
  `end_treatment_date`

### Group member roles (Progress Review I)

| # | Name | IT Number | Preprocessing technique | Notebook |
|---|------|-----------|--------------------------|----------|
| 1 | Kanakasekara K.A.D.T.B. | IT25101844 | Handling missing data | `notebooks/IT25101844_MissingData.ipynb` |
| 2 | Rukshidha S | IT25101824 | Encoding categorical variables | `notebooks/IT25101824_Encoding.ipynb` |
| 3 | Rukshani T.K.S. | IT25101852 | Outlier removal | `notebooks/IT25101852_OutlierRemoval.ipynb` |
| 4 | Wickramasinghe W.G.P.D. | IT25101868 | Normalization / scaling | `notebooks/IT25101868_Scaling.ipynb` |
| 5 | Edirinayake E.M.H.T. | IT25103394 | Feature engineering (creation, selection, PCA) | `notebooks/IT25103394_FeatureEngineering.ipynb` |
| 6 | Herath H.M.W.G.J.L. | IT25104015 | Duplicates, data types & class-imbalance check | `notebooks/IT25104015_DuplicatesDtypesImbalance.ipynb` |

### How to run the code (Google Colab)
1. Open [Google Drive](https://drive.google.com) and create the folder
   `data/raw/` in **My Drive** (or wherever you placed it).
2. Upload `Lung Cancer.csv` into that folder.
3. Open each notebook in Google Colab from Drive (double-click, or right-click
   → Open with → Google Colaboratory).
4. Run all cells top-to-bottom (`Runtime → Run all`). The first cell mounts
   Google Drive and asks you to authorize access — approve it once per session.
5. Each individual notebook is self-contained: it loads the raw CSV, applies
   one preprocessing technique, and produces an EDA visualization with
   interpretation.
6. `group_pipeline.ipynb` re-applies all six techniques in sequence and
   writes the final cleaned/encoded dataset to `results/outputs/`.

### Repository layout
```
Group_KU-22/
├── README.md
├── data/
│   ├── raw/                     # Assigned dataset as provided
│   └── external/                # Any external reference datasets (if used)
├── notebooks/
│   ├── IT25101844_MissingData.ipynb
│   ├── IT25101824_Encoding.ipynb
│   ├── IT25101852_OutlierRemoval.ipynb
│   ├── IT25101868_Scaling.ipynb
│   ├── IT25103394_FeatureEngineering.ipynb
│   └── IT25104015_DuplicatesDtypesImbalance.ipynb
├── group_pipeline.ipynb         # Integrated pipeline (combined work)
└── results/
    ├── eda_visualizations/      # Plots & charts (PNG/JPEG) exported from notebooks
    ├── logs/                    # Any execution logs (optional)
    └── outputs/                 # Final processed dataset / features
```

### Key EDA findings (for the viva)
- No missing values or duplicate rows were found in the raw data.
- `age`, `cancer_stage`, and `smoking_status` are each close to uniformly
  distributed across their categories/ranges.
- A small number of `age` outliers (~0.44% of rows) were the only outliers
  detected across the three numeric columns.
- The target class is imbalanced (~78% did not survive vs. ~22% survived) —
  plan to use F1-score, confusion matrix, and **stratified** k-fold cross
  validation in Step 4, not plain accuracy.
- Correlation between every numeric feature and `survived` is close to zero,
  which is worth discussing in the report's Ethical Considerations /
  Reflections sections.

### AI tool usage
See Section 7 (AI Tool Usage Declaration and Transparency) of the final
group report for full disclosure of how generative AI tools were used.
