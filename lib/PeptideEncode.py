
import numpy as np
import pandas as pd
import re
from tqdm import tqdm, trange



letterDict = {"A": 0,
              "C": 1,
              "D": 2,
              "E": 3,
              "F": 4,
              "G": 5,
              "H": 6,
              "I": 7,
              "K": 8,
              "L": 9,
              "M": 10,
              "N": 11,
              "P": 12,
              "Q": 13,
              "R": 14,
              "S": 15,
              "T": 16,
              "V": 17,
              "W": 18,
              "Y": 19,
              "U": 20,
              "B": 21}


def encodePeptideByInteger(peptide:str):
    vec = [letterDict[aa]+1 if aa in letterDict.keys() else 0 for aa in peptide]
    return vec

#letterDict = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11,
#                  "P": 12, "Q": 13, "R": 14, "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19, "U": 20, "B": 21}

def encodePeptideOneHot(peptide: str):

    AACategoryLen = len(letterDict)
    peptide_length = len(peptide)

    en_vector = np.zeros((peptide_length, AACategoryLen))

    i = 0
    for AA in peptide:
        if AA == "X":
            i = i + 1
            continue
        elif AA in letterDict.keys():
            en_vector[i][letterDict[AA]] = 1
            i = i + 1
        else:
            en_vector[i] = 0.5
            i = i + 1
            print("Error: invalid amino acid: %s" % (str(AA)))
            #exit(1)

    return en_vector

def encodePeptides(x):
    ## encoded data
    #n_aa_types = len(letterDict)
    #peptide_length = len(x['x'].iloc[0])
    #encoded_data = np.zeros((x.shape[0], peptide_length, n_aa_types))
    k = 0
    #for i, row in x.iterrows():
    #    peptide = row['x']
    #    encoded_data[k] = encodePeptideOneHot(peptide)
    #    k = k + 1

    encoded_data = np.array([encodePeptideOneHot(peptide) for peptide in x['x']])
    return encoded_data
