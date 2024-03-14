# DeepVEP
**DeepVEP**: mutation impact prediction on post-translational modifications using deep learning

### Table of contents:

- [DeepVEP](#deepvep)
    - [Table of contents:](#table-of-contents)
  - [Installation](#installation)
      - [Download DeepVEP](#download-deepvep)
      - [Installation](#installation-1)
      - [Download model files](#download-model-files)
  - [Usage](#usage)
      - [Mutation impact prediction:](#mutation-impact-prediction)
      - [PTM site prediction:](#ptm-site-prediction)
  - [How to cite:](#how-to-cite)

## Installation

#### Download DeepVEP

```shell
$ git clone https://github.com/bzhanglab/DeepVEP
```

#### Installation

DeepVEP is a python3 package. TensorFlow (>=2.6) is supported. Its dependencies can be installed via

```shell
$ pip install -r requirements.txt
```
DeepVEP has been tested on both Linux and Windows systems. It supports training and prediction on both CPU and GPU, but GPU is recommended for model training.

#### Download model files

The pretrained model files are available at [DeepVEP model repository](http://deepvep.ptmax.org/). After the model files are downloaded, decompress the *.tar.gz file and move all the files to the **models** folder as shown below:

```
├── README.md
├── deepvep.py
├── lib
│   ├── DataIO.py
│   ├── Metrics.py
│   ├── ModelView.py
│   ├── MutationUtils.py
│   ├── PTModels.py
│   ├── PeptideEncode.py
│   ├── RegCallback.py
│   ├── Utils.py
│   └── __init__.py
├── models
└── requirements.txt
```


## Usage

Run the following command line to show all command line options:
```
python deepmp.py predict -h
```

```
  -h or --help            show this help message and exit
  -i or --input           Input data for prediction
  -d or --db              Protein database
  -o or --out_dir         Output directory
  -w or --window_size     Window size for mutation impact prediction. In default, 7 amino acids in both side of a mutation.
  -m or --model           Trained model path
  -t or --task            Prediction type: 1=Mutation impact prediction, 2=PTM site prediction
  -e or --ensemble        Ensemble method, 1: average, 2: meta_lr, default is 1.
  -s, --explain_model     Perform model interpretability analysis
  -b or --bg_data         Data used as background data in model interpretability
```


#### Mutation impact prediction:

Below is an example for mutation impact prediction. The input for ``-i`` is a **TSV** format file which contains mutation information. The input for ``-d`` is a protein database file in FASTA format which contains the protein sequences for the wild type of proteins.

```
python deepmp.py predict -m models/ -i mutation_file.tsv -d protein.fasta -t 1 -o output_folder
```

The required columns for input of ``-i`` include **Protein**, **AA_Ref**, **AA_Pos** and **AA_Var**. An example is shown below:

```
Protein	AA_Ref	AA_Pos	AA_Var
P23246	I	    337	    A
```

For the above example input (``-i``), the wild type protein sequence of protein **P23246** should be present in the input protein database for ``-d``:
```
>P23246
MSRDRFRSRGGGGGGFHRRGGGGGRGGLHDFRSPPPGMGLNQNRGPMGPGPGQSGPKPPI
PPPPPHQQQQQPPPQQPPPQQPPPHQPPPHPQPHQQQQPPPPPQDSSKPVVAQGPGPAPG
VGSAPPASSSAPPATPPTSGAPPGSGPGPTPTPPPAVTSAPPGAPPPTPPSSGVPTTPPQ
AGGPPPPPAAVPGPGPGPKQGPGPGGPKGGKMPGGPKPGGGPGLSTPGGHPKPPHRGGGE
PRGGRQHHPPYHQQHHQGPPPGGPGGRSEEKISDSEGFKANLSLLRRPGEKTYTQRCRLF
VGNLPADITEDEFKRLFAKYGEPGEVFINKGKGFGFIKLESRALAEIAKAELDDTPMRGR
QLRVRFATHAAALSVRNLSPYVSNELLEEAFSQFGPIERAVVIVDDRGRSTGKGIVEFAS
KPAARKAFERCSEGVFLLTTTPRPVIVEPLEQLDDEDGLPEKLAQKNPMYQKERETPPRF
AQHGTFEYEYSQRWKSLDEMEKQQREQVEKNMKDAKDKLESEMEDAYHEHQANLLRQDLM
RRQEELRRMEELHNQEMQKRKEMQLRQEEERRRREEEMMIRQREMEEQMRRQREESYSRM
GYMDPRERDMRMGGGGAMNMGDPYGSGGQKFPPLGGGGGIGYEANPGVPPATMSGSMMGS
DMRTERFGQGGAGPVGGQGPRGMGPGTPAGYGRGREEYEGPNKKPRF
```



#### PTM site prediction:

Below is an example for PTM site prediction. The input (-d) is a protein database file in FASTA format which contains the protein sequences to predict.

```
python deepmp.py predict -m models/ -d protein.fasta -t 2 -o output_folder
```


## How to cite:

There is no a manuscript to cite yet.


