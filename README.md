# cs6910_assignment3

## Getting the code files
You need to first clone the github repository containing the files.
```
git clone https://github.com/pritamde090920/cs6910_assignment3.git
```
Then change into the code directory.
```
cd cs6910_assignment3
```
Make sure you are in the correct directory before proceeding further.


## Setting up the platform and environment
- ### Local machine
  If you are running the code on a local machine, then you need to have python installed in the machine and pip command added in the environemnt variables.
  You can execute the following command to setup the environment and install all the required packages
  ```
  pip install -r requirements.txt
  ```
- ### Google colab/Kaggle
  If you are using google colab platform or kaggle, then you need to execute the follwoing code
  ```
  pip install wandb argparse torch plotly matplotlib numpy pandas seaborn
  ```
This step will setup the environment required before proceeding.


## Project
The project deals in working with Sequence Learning Problems. It works with three types of cells:
- Recurrent Neural Networks (RNN)
- Long Short Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The project supports all the three types of cells with the encoder and decoder architectures. Additionally the feature of adding attention to the encoder outputs before passing them to the decoder is also supported.

## Loading the dataset
- Dataset is placed inside the project ```cs6910_assignment3``` directory
  - Target language is same as default language (Bengali)
    You do not need to do anything specifically for the dataset. The code will handle it automatically. You can simply run
    ```
    python train.py <any_specifications_related_to_the_model>
    ```
  - Target language is different
    You need to specify the path of the specific language directory inside the dataset. For example if you want to run on Telugu dataset the run it like :
    ```
    python train.py --root aksharantar_sampled/tel
    ```

- Dataset is placed outside the project directory
  In this case you need to specify the absolute path of the directory of the language (even if it is the default language Bengali) you want to run the model with.
  For example,
  ```
  python train.py --root <absolute_path_till_specific_language_directory_inside_aksharantar_sampled>
  ```

#### Note
Here aksharantar_sampled refers to the directory obtained after unzipping the ```aksharantar_sampled.zip``` provided along with the question.


## Training the model

To train the model, you need to compile and execute the [train.py](https://github.com/pritamde090920/cs6910_assignment3/blob/main/train.py) file, and pass additional arguments if and when necessary.\
It can be done by using the command:
```
python train.py
```
By the above command, the model will run with the default configuration.\
To customize the run, you need to specify the parameters like ```python train.py <*args>```\
For example,
```
python train.py -e 20 -b 128 --cell GRU
```
The arguments supported are :
|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | Pritam CS6910 - Assignment 3 | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     cs23m051    | Wandb Entity used to track experiments in the Weights & Biases dashboard |
|     `-r`,`--root`        |aksharantar_sampled/ben |Absolute path of the specific language in the dataset                                         |
|     `-e`, `--epochs`     |       15      | Number of epochs to train neural network                                 |
|   `-b`, `--batch`        |       64       | Batch size to divide the dataset                                  |
|   `-n`, `--neurons`        |       256       | Number of neurons in the fully connected layer                                  |
|   `-d`, `--dropout`        |       0.0       | Percentage of dropout in the network                                  |
|   `-em`, `--embedding`        |       256       | Size of the embedding layer                                  |
|   `-enc`, `--encoder`        |       3       | Number of layers in the encoder                                  |
|   `-dec`, `--decoder`        |       3       | Number of layers in the decoder                                  |
|   `-c`, `--cell`        |       LSTM       | Type of cell                                  |
|   `-bid`, `--bidir`        |       YES       | choices: [YES,NO]                                  |
|   `-t`, `--test`        |       0       | choices: [0,1]                                  |
|   `-att`, `--attention`        |       1       | choices: [0,1]                                  |
|   `-ht`, `--heat`        |       1       | choices: [0,1]                                  |
|   `-f`, `--font`        |       BengaliFont.TTF       | Font of the language chosen to generate the heatmap                                  |

The arguments can be changed as per requirement through the command line.
  - If prompted to enter the wandb login key, enter the key in the interactive command prompt.

## Testing the model
To test the model, you need to specify the test argument as 1. For example
```
python train.py -t 1
```
This will run the model with default parameters and print the test accuracy and loss.




## Additional features
The following features are also supported
  - If you need some clarification on the arguments to be passed, then you can do
    ```
    python train.py --help
    ```
  - If you want to apply attention and generate the 3x3 heatmap grid then you can run the model like below. This will save a png file in the same working directory and also log the image into the wandb project
    ```
    python train.py --attention 1 --heatmap 1
    ``` 
  - If you are using any language other than the default Bengali language set for the model, then to generate the heatmap you need to download and install the font of the language into your environment. You can find the required font in [Font](https://fonts.google.com). Here you can type in your preferred language and download the respective font file. Right click on the downloaded file and install it. Make sure you download the file into the same directory as the project. After successful installation, you need to specify the name of the .TTF file generated and run the model like :
    ```
    python train.py --font <name_of_the_font_file>.TTF
    ```
  
  

## Links
[Wandb Report](https://wandb.ai/cs23m051/Pritam%20CS6910%20-%20Assignment%203/reports/CS6910-Assignment-3-Pritam-De-CS23M051--Vmlldzo3OTc3MDY3)
