### LAC: Learning with Adversarial Domain Classifier to Enhance EEG Recognition Accuracy with Deep Convolution Network

This project explores the performance of various domain adaptation methods for electroencephalography (EEG) signal classification in a subject-dependent setting. The primary objective is to enhance the generalization capability of the models when dealing with unseen subjects.

#### Implemented Methods

We have implemented the following domain adaptation methods:

1. Adversarial Discriminative Domain Adaptation (ADDA)
2. Domain-Adversarial Training of Neural Networks (DANN)
3. Meta-Learning for Domain Generalization (MLDG)
4. MixStyle

#### Requirements

We construct and test the project on Windows 11 and Linux serve, the python version is `3.10`.
The required packages have been listed in the `requirements.txt`, you can use the followed commands to install them.
```
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ tllib==0.4
```

#### Usage

1. Clone the repository:
```
git clone https://github.com/Otsuts/TL-SEEDIV.git
cd TL-SEEDIV
```

2. Install the required dependencies:
```
pip install -r requirements.txt
pip install -i https://test.pypi.org/simple/ tllib==0.4
```


3. Run the main.py script with the desired options:
```
python main.py --model <model_name> --log <log_folder> --phase <phase> --learn_rate <learning_rate> --batch_size <batch_size> --num_epoch <num_epochs> --weight_decay <weight_decay> --patience <patience> --num_class <num_classes> --trade_off <trade_off> --pretrain_epoch <pretrain_epochs> --pretrain_lr <pretrain_learning_rate> --num_support <num_support> --num_domain <num_domains>
```

As we have use the `config.yaml` to save the best hyperparameters, if you want to use our saved hyperparameters. Please uncomment the followed code in the `main.py`
``` python
args = dict_to_namespace(config[args.model])
```

Then you can run the scipt simply using:
```
python main.py --model <model_name>
```


Replace `<model_name>` with the desired domain adaptation method (dann, adda, mixstyle, or mldg). Other options can be adjusted as needed.

4. Run the project

The script will output the subject-dependent accuracy for the selected domain adaptation method.

You can comment/uncomment the followed code to close/open the tsne visualization
``` python
tsne_visualize(self.test_data, label_pred, f'TL Prediction Labels ({self.args.model})')
```

## Results

Our experiments demonstrate the effectiveness of various domain adaptation methods for subject-dependent EEG signal classification. DANN outperforms all the other methods, achieving the highest classification accuracy. MLDG and MixStyle show similar performance, while ADDA achieves slightly lower accuracy.

