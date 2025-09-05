# DeepMVP
**DeepMVP**: mutation impact prediction on post-translational modifications using deep learning

### Table of contents:

- [DeepMVP](#deepmvp)
    - [Table of contents:](#table-of-contents)
  - [Installation](#installation)
      - [Download DeepMVP](#download-deepmvp)
      - [Installation](#installation-1)
      - [Download model files](#download-model-files)
  - [Usage](#usage)
      - [Mutation impact prediction:](#mutation-impact-prediction)
      - [PTM site prediction:](#ptm-site-prediction)
  - [How to cite:](#how-to-cite)

## Installation

#### Download DeepMVP

```shell
$ git clone https://github.com/bzhanglab/DeepMVP
```

#### Installation

DeepMVP is a python3 package. TensorFlow (>=2.6) is supported. Its dependencies can be installed via

```shell
$ pip install -r requirements.txt
```
DeepMVP has been tested on both Linux and Windows systems. It supports training and prediction on both CPU and GPU.

#### Download model files

The pretrained model files are available at [DeepMVP model repository](http://DeepMVP.ptmax.org/). After the model files are downloaded, decompress the *.tar.gz file and move all the files to the **models** folder as shown below:

```
├── README.md
├── DeepMVP.py
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
  -s or --explain_model   Perform model interpretability analysis
  -b or --bg_data         Data used as background data in model interpretability
```


#### Mutation impact prediction:

Below is an example for mutation impact prediction. The input for ``-i`` is a **TSV** format file which contains mutation information. The input for ``-d`` is a protein database file in FASTA format which contains the protein sequences for the wild type of proteins.

```
python DeepMVP.py predict -m models/ -i example/mutation_input.tsv -d example/Q5S007.fasta -t 1 -o mutation_output_folder
```

The required columns for input of ``-i`` include **Protein**, **AA_Ref**, **AA_Pos** and **AA_Var**. An example ("example/mutation_input.tsv") is shown below:

```
Protein  AA_Ref  AA_Pos  AA_Var
Q5S007   R       1441    C
Q5S007   R       1441    H
```

Below please find the description of each column in the output file "example/mutation_input.tsv":

| Column name  | Description |
| ------------ | ----------- |
| Protein | the protein name in the input fasta file |
| AA_Ref | the wild type amino acid |
| AA_Pos | the mutation position on the protein (1-based: the position of the first amino acid on the protein is 1) |
| AA_Var | the mutation amino acid |

An optional column `Variant_ID` can be added to the input. The same IDs would
be used in the output rather than generating IDs from row numbers.

For the above example input (``-i``), the wild type protein sequence of protein **Q5S007** should be present in the input protein database for ``-d``. The first 10 lines of the file "example/Q5S007.fasta" are shown below:

```
>Q5S007
MASGSCQGCEEDEETLKKLIVRLNNVQEGKQIETLVQILEDLLVFTYSERASKLFQGKNI
HVPLLIVLDSYMRVASVQQVGWSLLCKLIEVCPGTMQSLMGPQDVGNDWEVLGVHQLILK
MLTVHNASVNLSVIGLKTLDLLLTSGKITLLILDEESDIFMLIFDAMHSFPANDEVQKLG
CKALHVLFERVSEEQLTEFVENKDYMILLSALTNFKDEEEIVLHVLHCLHSLAIPCNNVE
VLMSGNVRCYNIVVEAMKAFPMSERIQEVSCCLLHRLTLGNFFNILVLNEVHEFVVKAVQ
QYPENAALQISALSCLALLTETIFLNQDLEEKNENQENDDEGEEDKLFWLEACYKALTWH
RKNKHVQEAACWALNNLLMYQNSLHEKIGDEDGHFPAHREVMLSMLMHSSSKEVFQASAN
ALSTLLEQNVNFRKILLSKGIHLNVLELMQKHIHSPEVAESGCKMLNHLFEGSNTSLDIM
AAVVPKILTVMKRHETSLPVQLEALRAILHFIVPGMPEESREDTEFHHKLNMVKKQCFKN
```

The output folder ("mutation_output_folder") of the example command line looks like below:

```
mutation_output_folder
├── deepmvp-mutation_impact.tsv
├── acetylation_k
├── glycosylation_n
├── methylation_k
├── methylation_r
├── phosphorylation_st
├── phosphorylation_y
├── sumoylation_k
└── ubiquitination_k
```

The output file "mutation_output_folder/deepmvp-mutation_impact.tsv" contains the predicted mutation impact on all the PTM sites supported by DeepMVP. This is the only file that users need to use for downstream analysis. The other folders contain intermediate prediction files.

```
Protein  AA_Ref  AA_Pos  AA_Var  pos   diff_pos  w_pep            m_pep            w_prob      m_prob      delta_prob   ptm
Q5S007   R       1441    C       1443  -2        FNIKARASSSPVILV  FNIKACASSSPVILV  0.7837325   0.2552052   -0.5285273   phosphorylation_st
Q5S007   R       1441    C       1444  -3        NIKARASSSPVILVG  NIKACASSSPVILVG  0.88949174  0.43282443  -0.45666731  phosphorylation_st
Q5S007   R       1441    C       1445  -4        IKARASSSPVILVGT  IKACASSSPVILVGT  0.8771168   0.49486965  -0.38224715  phosphorylation_st
Q5S007   R       1441    H       1443  -2        FNIKARASSSPVILV  FNIKAHASSSPVILV  0.7837325   0.26106784  -0.52266466  phosphorylation_st
Q5S007   R       1441    H       1444  -3        NIKARASSSPVILVG  NIKAHASSSPVILVG  0.88949174  0.44303632  -0.44645542  phosphorylation_st
Q5S007   R       1441    H       1445  -4        IKARASSSPVILVGT  IKAHASSSPVILVGT  0.8771168   0.48524436  -0.39187244  phosphorylation_st
```

Below please find the description of each column in the output file "output_folder/site_prediction.tsv":

| Column name  | Description |
| ------------ | ----------- |
| Protein | the protein name in the input fasta file |
| AA_Ref | the wild type amino acid |
| AA_Pos | the mutation position on the protein (1-based: the position of the first amino acid on the protein is 1) |
| AA_Var | the mutation amino acid |
| pos | the PTM site on the protein (1-based: the position of the first amino acid on the protein is 1) |
| diff_pos | the distance between the PTM site and the mutation site |
| w_pep | the wild type peptide sequence in which the center is PTM site|
| m_pep | the mutant peptide sequence in which the center is PTM site|
| w_prob | predicted PTM site probability for wild type sequence|
| m_prob | predicted PTM site probability for mutant sequence|
| delta_prob | mutation impact on the PTM site: m_prob - w_prob|
| ptm | PTM name |


The prediction took less than 3 minutes using CPU on a Linux server (64G RAM and 16 CPUs).


#### PTM site prediction:

Below is an example for PTM site prediction. The input (-d) is a protein database file in FASTA format which contains the protein sequences to predict. The following command line is used to predict all PTM sites supported by DeepMVP.

```
python DeepMVP.py predict -m models/ -d example/Q5S007.fasta -t 2 -o output_folder
```

The output folder ("output_folder") of the example command line looks like below:

```
output_folder/
├── site_prediction.tsv
├── acetylation_k
├── glycosylation_n
├── methylation_k
├── methylation_r
├── phosphorylation_st
├── phosphorylation_y
├── sumoylation_k
└── ubiquitination_k
```

The output file "output_folder/site_prediction.tsv" contains all the predicted PTM sites. This is the only file that users need to use for downstream analysis. The other folders contain intermediate prediction files. 

```
protein  aa  pos   x                                y_pred          fpr                 ptm
Q5S007   K   17    ASGSCQGCEEDEETLKKLIVRLNNVQEGKQI  0.99253386      0.0027506112469437  acetylation_k
Q5S007   K   18    SGSCQGCEEDEETLKKLIVRLNNVQEGKQIE  0.54781103      0.1937652811735941  acetylation_k
Q5S007   K   30    TLKKLIVRLNNVQEGKQIETLVQILEDLLVF  0.0013190061    0.9529339853300732  acetylation_k
Q5S007   K   53    ILEDLLVFTYSERASKLFQGKNIHVPLLIVL  0.70806825      0.1253056234718826  acetylation_k
Q5S007   K   58    LVFTYSERASKLFQGKNIHVPLLIVLDSYMR  0.0148314405    0.7331907090464548  acetylation_k
Q5S007   K   87    MRVASVQQVGWSLLCKLIEVCPGTMQSLMGP  0.0005026354    0.9819682151589242  acetylation_k
Q5S007   K   120   VGNDWEVLGVHQLILKMLTVHNASVNLSVIG  0.00056051335   0.9801344743276283  acetylation_k
Q5S007   K   137   LTVHNASVNLSVIGLKTLDLLLTSGKITLLI  0.007324457     0.82059902200489    acetylation_k
Q5S007   K   147   SVIGLKTLDLLLTSGKITLLILDEESDIFML  0.0003533643    0.9865525672371638  acetylation_k
```

Below please find the description of each column in the output file "output_folder/site_prediction.tsv":

| Column name  | Description |
| ------------ | ----------- |
| protein | the protein name from the input fasta file |
| aa | the amino acid PTM site to predict |
| pos | the PTM site on the protein (1-based: the position of the first amino acid on the protein is 1) |
| x | a peptide sequence of 31 amino acids in which the center is the predicted PTM site |
| y_pred | predicted probability |
|fpr| false positive rate using the predicted probability as the threshold to define positive PTM site |
| ptm | PTM name |


The prediction took less than 3 minutes using CPU on a Linux server (64G RAM and 16 CPUs).

## Protein Consequences

Often, variants would be otained as changes in coding sequences. These would
need to be translated to the corresponding change in protein sequence. 

Below is an example of two varaints in VCF format.

```
head example/mutation_input.vcf 
12      40310434                C       T
12      40310435                G       A
```

Variant can be translated to protein consequence and outputed in the format 
required by DeepMVP. `translate` calls [ProtVar API](https://www.ebi.ac.uk/ProtVar/)
therefore a connection to the internet is required for it to work

```
python DeepMVP.py translate -i example/mutation_input.vcf
Variant_ID  Protein AA_Ref  AA_Pos  AA_Var
chr12:40310434:C:T  Q5S007  R       1441    C
chr12:40310435:G:A  Q5S007  R       1441    H
```
In addition, the FASTA (homo sapians) file which contains the protein sequences
for the wild type of all proteins can be downloaed and edited as follows

```
wget wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
zcat ref/uniprot_sprot.fasta.gz | awk '/^>/ {split($0, a, "|"); print ">" a[2]; next} {print}' > ref/uniprot_sprot_modified.fasta
grep -A 3 Q5S007 uniprot_sprot_modified.fasta 
>Q5S007
MASGSCQGCEEDEETLKKLIVRLNNVQEGKQIETLVQILEDLLVFTYSERASKLFQGKNI
HVPLLIVLDSYMRVASVQQVGWSLLCKLIEVCPGTMQSLMGPQDVGNDWEVLGVHQLILK
MLTVHNASVNLSVIGLKTLDLLLTSGKITLLILDEESDIFMLIFDAMHSFPANDEVQKLG
```


## How to cite:

Wen, B., Wang, C., Li, K. et al. DeepMVP: deep learning models trained on high-quality data accurately predict PTM sites and variant-induced alterations. **Nat Methods** (2025). https://doi.org/10.1038/s41592-025-02797-x


