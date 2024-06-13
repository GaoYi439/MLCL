# MLCL
This code gives the implementation  of the paper "Complementary to Multiple Labels: A Correlation-Aware Correction Approach".

 Requirements
- Python >=3.6
- PyTorch >=1.10

---
main.py
>This is main function. Our approach is a two-stage strategy, including estimating the transition matrix and training a multi-labeled classifier. The first stage is to estimate transition matrix, and the second stage is to learn a multi-labeled classifier. When running the first stage, you should save the trained model about predicting complementary labels, and then estimate the transition matrix. The estimated transition matrix will be saved in the directory.
>
>According the transition matrix estimated in the first stage, we can train a multi-labeled classifier in the second stage. After running the second stage, you should see a .csv file with the results saved in the directory. The results will have seven columns: epoch number, training loss, hamming loss of test data, one error of test data, coverage of test data, ranking loss of test data and average precision of test data.

generate.py
>This is used to generate complementary labels and select a relevant label from labeled data (in section 5.2). After running, you should see a .csv file of complementary labels or relevant labels for a dataset in the vector form. If you have prepared the training data and its complementary labels or relevant labels, please ignore it.

---
## Running the first stage

Training a model to predict complementary labels, which will be saved and used to calculate an initial transition matrix S (Eq. (6)) in the following steps.

python main.py --stage first --step one --dataset \<dataset name\>

Estimating the transition matrix with label correlation matrix.

python main.py --stage first --step two --dataset \<dataset name\>

## Running the second stage

python main.py --stage second --dataset \<dataset name\>

## Running with relevant labels
python main.py --stage second --one_label True --dataset \<dataset name\>

Specify the dataset argument:
- scene: scene dataset
- yeast: yeast dataset
