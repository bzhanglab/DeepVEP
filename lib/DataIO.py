from numpy.random import seed

seed(2019)
# from tensorflow import set_random_seed
# set_random_seed(2020)
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("agg")

from .PeptideEncode import *

from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import Counter
from pyteomics import parser
import random
import subprocess
import re
import os
import json
import sys
import multiprocessing as mp
from sklearn.utils import shuffle
from math import ceil, floor


def data_processing_range(train_file: str, test_file=None, db=None, use_all_data=True, flank_length_range=None, out_dir="./",
                          random_seed=2018,seq_encode_method="one_hot"):
    proMap = dict()
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proSeq = proSeq.replace('#', '')
        proMap[record.id] = proSeq

    data = dict()

    if flank_length_range is None:
        flank_length_range = list(range(15, 30))
    for flank_length in flank_length_range:
        siteData = pd.read_table(train_file, sep="\t", header=0, low_memory=False)
        siteData['x'] = [getPeptideSequence(proMap[protein],pos,flank_length) for protein, pos in zip(siteData['protein'], siteData['pos'])]
        ## output
        out_train_file = out_dir + "/train_data_" + str(flank_length) + ".tsv"
        siteData.to_csv(out_train_file,sep="\t",header=True,index=False)
        if test_file is None:
            out_test_file = None
        else:
            test_site_data = pd.read_table(test_file, sep="\t", header=0, low_memory=False)
            test_site_data['x'] = [getPeptideSequence(proMap[protein], pos, flank_length) for protein, pos in zip(test_site_data['protein'], test_site_data['pos'])]
            out_test_file = out_dir + "/test_data_" + str(flank_length) + ".tsv"
            test_site_data.to_csv(out_test_file, sep="\t", header=True, index=False)

        X_train, Y_train, X_test, Y_test = data_processing(out_train_file, out_test_file, use_all_data = use_all_data, out_dir = out_dir, random_seed = random_seed,
                                                           seq_encode_method=seq_encode_method)
        max_length = 2*flank_length+1
        data[max_length] = [X_train, Y_train, X_test, Y_test]
    return data

def data_processing(train_file: str, test_file=None, use_all_data=True, out_dir="./",random_seed=2018,seq_encode_method="one_hot"):
    """
    PTM site data processing for training
    :param input_data:
    :param test_file:
    :param out_dir:
    :param random_seed:
    :return:
    """
    siteData = pd.read_table(train_file, sep="\t", header=0, low_memory=False)
    peptide_length = len(siteData['x'].iloc[0])
    print("Peptide length: %d!" % (peptide_length))

    ## Shuffle the input data
    # siteData = siteData.sample(siteData.shape[0], replace=False, random_state=2018)
    siteData = shuffle(siteData, random_state=random_seed)
    siteData = siteData.reset_index(drop=True)

    #n_aa_types = len(letterDict)

    # train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2]))

    ## need to count the number of samples for each group
    siteData_1 = siteData.query('y==1')
    siteData_0 = siteData.query('y==0')

    if seq_encode_method == "one_hot":
        encoded_data_1 = encodePeptides(siteData_1)
        encoded_data_0 = encodePeptides(siteData_0)
    elif seq_encode_method == "embedding":
        ## TODO
        print("embedding is not supported yet!")
        sys.exit(1)
    else:
        print("Invalid sequence encoding method: %s" % (seq_encode_method))
        sys.exit(1)

    n_1 = encoded_data_1.shape[0]
    n_0 = encoded_data_0.shape[0]

    if n_0 > n_1:
        print("Samples are not balanced in classes: y==1 => %d, y==0 => %d !" % (n_1, n_0))
    else:
        print("Samples are balanced in classes: y==1 => %d, y==0 => %d !" % (n_1, n_0))

    Y_label = np.array([1, 0])

    X_test = np.empty(1)
    Y_test = np.empty(1)

    if test_file is None:
        n_val = ceil(encoded_data_1.shape[0] * 0.1)
        X_test = np.append(encoded_data_1[0:n_val], encoded_data_0[0:n_val], axis=0)
        X_test = X_test.astype('float32')
        # Y_test = to_categorical(np.repeat(Y_label, [n_val, n_val], axis=0), num_classes=2)
        Y_test = np.repeat(Y_label, [n_val, n_val], axis=0)

        encoded_data_1 = encoded_data_1[n_val:]
        encoded_data_0 = encoded_data_0[n_val:]

        n_1 = encoded_data_1.shape[0]
        n_0 = encoded_data_0.shape[0]

    else:
        print("Use data in file %s for testing !" % (test_file))
        test_data = pd.read_table(test_file, sep="\t", header=0, low_memory=False)


        if seq_encode_method == "one_hot":
            X_test = encodePeptides(test_data)
        elif seq_encode_method == "embedding":
            ## TODO
            print("embedding is not supported yet!")
            sys.exit(1)
        else:
            print("Invalid sequence encoding method: %s" % (seq_encode_method))
            sys.exit(1)


        X_test = X_test.astype('float32')
        Y_test = test_data['y'].to_numpy()

    if use_all_data:
        if n_0 > n_1:

            X_train = list()
            Y_train = list()
            n_iteration = floor(1.0 * n_0 / n_1)
            print("Total iteration:", n_iteration, "\n")

            last_index_of_neg = 0
            for i in range(int(n_iteration)):
                print("\nIteration: ", i)
                new_index = last_index_of_neg + n_1
                X_train_i = np.append(encoded_data_1, encoded_data_0[last_index_of_neg:new_index], axis=0).astype(
                    "float32")
                Y_train_i = np.repeat(Y_label, [n_1, n_1], axis=0)
                X_train.append(X_train_i)
                Y_train.append(Y_train_i)
                last_index_of_neg = new_index

        else:
            ## use what we have
            X_train = np.append(encoded_data_1, encoded_data_0, axis=0).astype("float32")
            Y_train = np.repeat(Y_label, [n_1, n_0], axis=0)

    else:
        if n_0 > n_1:
            ## balanced samples
            X_train = np.append(encoded_data_1, encoded_data_0[0:n_1], axis=0).astype("float32")
            Y_train = np.repeat(Y_label, [n_1, n_1], axis=0)

        else:
            ## use what we have
            X_train = np.append(encoded_data_1, encoded_data_0, axis=0).astype("float32")
            Y_train = np.repeat(Y_label, [n_1, n_0], axis=0)

    return [X_train, Y_train, X_test, Y_test]


def processing_prediction_data(input_data: str, db:str, flank_length:int):

    proMap = dict()
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proSeq = proSeq.replace('#', '')
        proMap[record.id] = proSeq

    siteData = pd.read_table(input_data, sep="\t", header=0, low_memory=False)

    siteData['x'] = [getPeptideSequence(proMap[protein], pos, flank_length) for protein, pos in
                     zip(siteData['protein'], siteData['pos'])]

    #if out_dir is not None:
    #    siteData.to_csv(out_dir + "/x_" +str(flank_length) +".tsv",sep="\t",header=True,index=False)

    x = encodePeptides(siteData)

    x = x.reshape((x.shape[0], x.shape[1], x.shape[2]))

    x = x.astype('float32')

    return [x,siteData]


def generate_sample_specific_training_data(mod_file=None, ptm_type="sty", header=True, db=None, flank_length=15,
                                           split_ratio = 0.1,
                                           enzyme="trypsin",
                                           max_missed_cleavages=2, min_peptide_length=7, max_peptide_length=45,
                                           min_n = 10, # only consider negative sites from those proteins who have >= min_n known sites
                                           out_dir="./", refine_file=None):

    # read modification data
    siteHeader = None
    if (header == True):
        siteHeader = 0
    siteData = pd.read_table(mod_file, sep="\t", header=siteHeader)
    print("The number of modification sites in file %s : %d" % (mod_file, siteData.shape[0]))
    proteins = dict()
    for i, row in siteData.iterrows():
        proID = row['protein']
        mod_aa = row['aa']
        if "pos" in siteData.columns:
            pos = row['pos']
        else:
            pos = row['position']
        if proID not in proteins.keys():
            proteins[proID] = {pos: mod_aa}
        else:
            proteins[proID][pos] = mod_aa

    print("Proteins in file: %s : %d" % (mod_file, len(proteins)))
    out_file = out_dir + "/all_pos_neg_samples.tsv"
    getAllModificationSites(db=db, out_file=out_file, ptm_type=ptm_type, flank_length=flank_length, proteins=proteins,
                            enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                            min_peptide_length=min_peptide_length, max_peptide_length=max_peptide_length)
    siteData = pd.read_table(out_file, sep="\t", header=0)
    print(Counter(siteData['y']))
    final_file = out_dir + "/all_pos_neg_samples_rm_redundant.tsv"
    removeRedundantSite(out_file, outfile=final_file)
    siteData = pd.read_table(final_file, sep="\t", header=0)


    ## filter negative samples
    siteData = filter_negative_samples(siteData, min_n=min_n)

    ## refine negative samples
    if refine_file is not None:
        siteData = refine_negative_samples(siteData, refine_file=refine_file)

    ## save proteins
    out_db = out_dir + "/protein.fasta"
    get_proteins(mod_file, db, out_db)

    print(Counter(siteData['y']))

    siteData.to_csv(out_dir + "/valid_pos_neg_samples.tsv", header=True, sep="\t", index=False)

    if split_ratio > 0:
        ## balanced splitting
        ## need to count the number of samples for each group
        siteData = siteData.sample(frac=1,random_state=2020).reset_index(drop=True)
        siteData_1 = siteData.query('y==1')
        siteData_0 = siteData.query('y==0')
        n_1 = siteData_1.shape[0]
        n_0 = siteData_0.shape[0]
        n_val = ceil(n_1 * 0.1)
        test_data = pd.concat([siteData_1[0:n_val],siteData_0[0:n_val]],axis=0)
        train_data = pd.concat([siteData_1[n_val:],siteData_0[n_val:]],axis=0)
        print("Training data:\n")
        print(Counter(train_data['y']))

        print("Testing data:\n")
        print(Counter(test_data['y']))

        train_data.to_csv(out_dir + "/train_data.tsv", header=True, sep="\t", index=False)
        test_data.to_csv(out_dir + "/test_data.tsv", header=True, sep="\t", index=False)

def filter_negative_samples(siteData, min_n=10):

    print("For negative samples, only use the sites from the proteins with >= %d known sites." % (min_n))
    siteData_1 = siteData.query('y==1')
    siteData_0 = siteData.query('y==0')

    pro = siteData_1.groupby('protein').size().reset_index(name='counts').query('counts>=@min_n')
    res = siteData_0[siteData_0['protein'].isin(pro['protein'])].copy(deep=True).reset_index(drop=True)
    test_data = pd.concat([siteData_1,res], axis=0)
    final_data = test_data.reset_index(drop=True)
    return final_data

def refine_negative_samples(siteData, refine_file=None):

    # refine_file may contain multiple file names
    print("Refine negative samples by removing any sites in file(s): %s" % (refine_file))

    siteData_1 = siteData.query('y==1').reset_index(drop=True)
    siteData_0 = siteData.query('y==0').reset_index(drop=True)
    file_list = refine_file.split(",")

    siteData_0['protein_aa_position'] = siteData_0.loc[:,('protein')] + "_" + siteData_0.loc[:,('aa')] + "_" + siteData_0.loc[:,('pos')].astype(str)
    n_rows_0 = siteData_0.shape[0]

    for f in file_list:
        a = pd.read_table(f,sep="\t",header=0)
        if 'pos' in a.columns:
            a['protein_aa_position'] = a.loc[:,('protein')] + "_" + a.loc[:,('aa')] + "_" + a.loc[:,('pos')].astype(str)
        else:
            a['protein_aa_position'] = a.loc[:,('protein')] + "_" + a.loc[:,('aa')] + "_" + a.loc[:,('position')].astype(str)

        tmp = siteData_0[~siteData_0['protein_aa_position'].isin(a['protein_aa_position'])]
        n_rm_rows = siteData_0.shape[0] - tmp.shape[0]
        print("Removed %d (%d) negative sites from file: %s" % (n_rm_rows, siteData_0.shape[0], f))
        siteData_0 = tmp.reset_index(drop=True)

    n_rm_rows = n_rows_0 - siteData_0.shape[0]
    print("In total, removed %d out of %d negative sites from file: %s" % (n_rm_rows, n_rows_0, refine_file))

    siteData_0_new = siteData_0.drop(columns=['protein_aa_position'])

    test_data = pd.concat([siteData_1,siteData_0_new], axis=0)
    final_data = test_data.reset_index(drop=True)
    return final_data

def get_proteins(file: str, db: str, out_file: str):
    siteData = pd.read_table(file, sep="\t")
    proteins = dict()
    for i, row in siteData.iterrows():
        proID = row['protein']
        proteins[proID] = 1

    records = list()
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        if proteins is not None:
            if record.id in proteins.keys():
                records.append(record)

    with open(out_file, "w") as output_handle:
        SeqIO.write(records, output_handle, "fasta")

    print("Save protein sequences to file %s" % (out_file))


def importData(site_file, db, out_file="site.txt", header=True, flank_length=15):
    '''
    Import site data.
    :param site_file: A tsv format file contains protein ID, site position (1-based) and amino acid
    :param db: Protein database in FASTA format
    :param out_file: Output file (x: peptide, y: label[1,0])
    :param header:
    :param flank_length:
    :return:
    '''

    print("Read modification data from file: " + site_file + " and " + db + ".\n")

    #
    proMap = {}
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proSeq = proSeq.replace('#', '')
        proMap[record.id] = proSeq
        # print(record.id)

    # read modification data
    siteHeader = None
    if (header == True):
        siteHeader = 1
    siteData = pd.read_table(site_file, sep="\t", header=siteHeader)
    print("The number of modification sites in file %s : %d" % (site_file, siteData.shape[0]))

    # sitefile:
    # NP_001273540.1  S   272

    ofile = open(out_file, "w")

    for i, row in siteData.iterrows():
        proID = row[0]
        pos = row[2]
        flankSeq = getPeptideSequence(proMap[proID], pos, length=flank_length)
        ofile.write(proID + "\t" + str(pos) + "\t" + row[1] + "\t" + flankSeq + "\n")

    ofile.close()
    return ([siteData, proMap])


def getPeptideSequence(proteinSequence: str, pos, length=15):
    """
    Get peptide sequence according to a position
    :param proteinSequence:
    :param pos:
    :param length:
    :return:
    """
    pSeq = proteinSequence.center(len(proteinSequence) + length * 2, 'X')
    newPos = pos + length
    startPos = pos - 1
    endPos = newPos + length
    flankSeq = pSeq[startPos:endPos]
    return flankSeq


def getAAcodingMap():
    aaMap = {}

    aaSeq = "XRKDEQNHSTYCWAILMFVPGUB"
    aalen = len(aaSeq)
    n = 0
    for i in range(0, aalen):
        for j in range(0, aalen):
            # print(aaSeq[i]+aaSeq[j])
            aa = aaSeq[i] + aaSeq[j]
            aaMap[aa] = n
            n = n + 1

    return (aaMap)


def getAllModificationSites(db: str, out_file: str, ptm_type="sty", flank_length=15, proteins=None, enzyme=None,
                            max_missed_cleavages=2,
                            min_peptide_length=7, max_peptide_length=45):
    """
    This is used to prepare training data for sample specific modeling
    :param db:
    :param ofile:
    :param length:
    :param proteins: a dict object, protein ID : 1-based position
    :param enzyme:
    :param max_missed_cleavages:
    :param min_peptide_length:
    :param max_peptide_length:
    :return:
    """
    print("Database file: %s." % (db))
    # ptmMap = {'S':1,'T':1,'Y':1}
    ptmMap = {}
    for aa in ptm_type:
        ptmMap[aa.upper()] = 1

    ofile = open(out_file, "w")
    if proteins is not None:
        if enzyme is not None:
            ofile.write("protein\taa\tpos\tx\ty\tpeptide\n")
        else:
            ofile.write("protein\taa\tpos\tx\ty\n")
    else:
        if enzyme is not None:
            ofile.write("protein\taa\tpos\tx\tpeptide\n")
        else:
            ofile.write("protein\taa\tpos\tx\n")

    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        if proteins is not None:
            if record.id not in proteins.keys():
                ## only consider protein which is exist in proteins
                print(record.id + " don't have sites.")
                continue

        proSeq = str(record.seq)
        proSeq = proSeq.replace('#', '')

        pos2peptide = dict()
        if enzyme is not None:
            pSet = parser.cleave(proSeq, parser.expasy_rules[enzyme],
                                 max_missed_cleavages,
                                 min_length=min_peptide_length)
            for pep in pSet:
                if (len(pep) <= max_peptide_length) and (len(pep) >= min_peptide_length):
                    # a = "aaabbbcccddd"
                    # a.index("bbb")
                    # 3
                    start_pos = proSeq.index(pep) + 1
                    end_pos = start_pos + len(pep) - 1
                    for pp in range(start_pos, end_pos + 1):
                        if pp not in pos2peptide:
                            pos2peptide[pp] = [pep]
                        else:
                            pos2peptide[pp].append(pep)

        for i in range(0, len(proSeq)):
            if (proSeq[i] in ptmMap.keys()):
                pos = i + 1
                flankSeq = getPeptideSequence(proSeq, pos, length=flank_length)
                if proteins is not None:
                    if pos in proteins[record.id].keys():
                        y = 1
                    else:
                        y = 0
                    if enzyme is not None:
                        if pos in pos2peptide.keys():
                            peptide = ";".join(pos2peptide[pos])
                        else:
                            peptide = "-"
                        ofile.write(
                            record.id + "\t" + proSeq[i] + "\t" + str(pos) + "\t" + flankSeq + "\t" + str(
                                y) + "\t" + peptide + "\n")
                    else:
                        ofile.write(
                            record.id + "\t" + proSeq[i] + "\t" + str(pos) + "\t" + flankSeq + "\t" + str(y) + "\n")
                else:
                    if enzyme is not None:
                        if pos in pos2peptide.keys():
                            peptide = ";".join(pos2peptide[pos])
                        else:
                            peptide = "-"
                        ofile.write(
                            record.id + "\t" + proSeq[i] + "\t" + str(pos) + "\t" + flankSeq + "\t" + peptide + "\n")
                    else:
                        ofile.write(record.id + "\t" + proSeq[i] + "\t" + str(pos) + "\t" + flankSeq + "\n")

    ofile.close()


def getSiteDataFromUniProt(db:str, gff:str, ptm_type = "phosphorylation", out_dir="./"):
    '''
    Exact PTM information from UniProt database
    :param db:
    :param gff:
    :param out_dir:
    :return:
    '''
    pro2seq = dict()

    # Create target Directory if don't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    output_fa = out_dir + "/input.fasta"
    ofa = open(output_fa, "w")

    for record in SeqIO.parse(db, "fasta"):
        # print(record.id)
        pro_ids = record.id.split("|")
        pro = pro_ids[1]
        # print(pro)
        pro2seq[pro] = record.seq
        ofa.write(">" + pro + "\n" + str(record.seq) + "\n")

    print("Protein sequences in db %s: %d" % (db, len(pro2seq)))

    output_table = out_dir + "/input_uniprot.txt"

    otable = open(output_table, "w")
    otable.write("protein\taa\tpos\n")

    ## read gff
    with open(gff) as f:
        for line in f:
            if line.startswith("#"):
                continue
            else:
                d = line.split("\t")
                pro = d[0]
                site_type = d[2]
                pos1 = int(d[3])
                pos2 = int(d[4])
                evidence = d[8]

                target_ptm = False

                if ptm_type == "phosphorylation" and site_type == "Modified residue":
                    if "Note=Phosphotyrosine" in evidence:
                        aa = "Y"
                        target_ptm = True
                    elif "Note=Phosphoserine" in evidence:
                        aa = "S"
                        target_ptm = True
                    elif "Note=Phosphothreonine" in evidence:
                        aa = "T"
                        target_ptm = True
                    else:
                        target_ptm = False

                if target_ptm:
                    if pos1 == pos2:
                        if aa == pro2seq[pro][pos1 - 1]:
                            otable.write(pro + "\t" + pro2seq[pro][pos1 - 1] + "\t" + str(pos1) + "\n")
                        else:
                            print("Amino acid doesn't match: %s => %s : %s => %s" % (aa, pro2seq[pro][pos1 - 1], line, pro2seq[pro]))
                        #if "ECO:0000269" in evidence and "asparagine" in evidence:

                    else:
                        print(line)

                #if site_type == "Glycosylation":
                #    ## Glycosylation
                #    if pos1 == pos2:
                #        if "ECO:0000269" in evidence and "asparagine" in evidence:
                #            otable.write(pro + "\t" + pro2seq[pro][pos1 - 1] + "\t" + str(pos1) + "\n")
                #    else:
                #        print(line)

    otable.close()


def getTrainDataFromFasta(db: str, ptm_type="sty", out_dir="./", flank_length=15, enzyme=None, max_missed_cleavages=2,
                          min_peptide_length=7, max_peptide_length=45, min_n=10):
    """
    Prepare modification for general model training from MusiteDeep fasta file
    :param db:
    :param ofile:
    :return:
    """
    out_file = out_dir + "/site_data.txt"
    out_db = out_dir + "/db.fasta"
    o_file = open(out_file, "w")
    db_file = open(out_db, "w")
    o_file.write("protein\taa\tpos\tx\ty\n")

    ptmMap = {}
    for aa in ptm_type:
        # need uppercase
        ptmMap[aa.upper()] = 1

    # ptmMap = {'S': 1, 'T': 1, 'Y': 1}

    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        newSeq = proSeq.replace('#', '')
        allSite = {}
        db_file.write(">" + record.id + "\n" + newSeq + "\n")
        k = 0
        # print("length:",len(proSeq)," ",len(newSeq))
        for i in range(0, len(proSeq)):
            # print(i,proSeq[i])
            if (proSeq[i] == '#'):
                k = k + 1
                pos = (i + 1) - k
                allSite[pos] = 1
                # print("pos:",pos)

        for i in range(0, len(newSeq)):
            if (newSeq[i] in ptmMap.keys()):
                pos = i + 1
                if (pos not in allSite.keys()):
                    allSite[pos] = 0
                flankSeq = getPeptideSequence(newSeq, pos, length=flank_length)
                if allSite[pos] == 1:
                    o_file.write(record.id + "\t" + newSeq[i] + "\t" + str(pos) + "\t" + flankSeq + "\t" + str(
                        allSite[pos]) + "\n")

    o_file.close()
    db_file.close()
    siteData = pd.read_table(out_file, sep="\t", header=0)
    print("Summary: ", out_file, " => ", Counter(siteData['y']))

    if enzyme is not None:
        generate_sample_specific_training_data(mod_file=out_file, header=True, db=out_db, flank_length=flank_length,
                                               enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                                               min_peptide_length=min_peptide_length,
                                               max_peptide_length=max_peptide_length,
                                               min_n = min_n,
                                               out_dir=out_dir)


def getTrainDataFromPhosphoSitePlusFasta(db: str, ptm_type="sty", out_dir="./", flank_length=15, enzyme=None,
                                         max_missed_cleavages=2,
                                         min_peptide_length=7, max_peptide_length=45, min_n=10):
    """
    Prepare modification for general model training from PhosphoSitePlus fasta file
    :param db: Phosphosite_PTM_seq.fasta, the lowercase site is modification site.
    :param ofile:
    :return:
    """

    out_file = out_dir + "/site_data.txt"
    out_db = out_dir + "/db.fasta"
    o_file = open(out_file, "w")
    db_file = open(out_db, "w")
    o_file.write("protein\taa\tpos\tx\ty\n")

    ptmMap = {}
    ptmMap_upper = {}
    for aa in ptm_type:
        ptmMap[aa.lower()] = 1
        ptmMap_upper[aa.upper()] = 1

    # ptmMap = {'s': 1, 't': 1, 'y': 1}
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        ## Please note the phosphosites in the fasta files are lowercase.
        proSeq = str(record.seq)
        newSeq = proSeq.upper()
        allSite = {}
        db_file.write(">" + record.id + "\n" + newSeq + "\n")
        # print(proSeq)

        for i in range(0, len(proSeq)):
            if (proSeq[i] in ptmMap.keys()):
                pos = i + 1
                allSite[pos] = 1

        # ptmMap_upper = {'S': 1, 'T': 1, 'Y': 1}
        for i in range(0, len(newSeq)):
            if (newSeq[i] in ptmMap_upper.keys()):
                pos = i + 1
                if (pos not in allSite.keys()):
                    allSite[pos] = 0
                flankSeq = getPeptideSequence(newSeq, pos, length=flank_length)
                if allSite[pos] == 1:
                    o_file.write(record.id + "\t" + newSeq[i] + "\t" + str(pos) + "\t" + flankSeq + "\t" + str(
                        allSite[pos]) + "\n")

    o_file.close()
    db_file.close()
    siteData = pd.read_table(out_file, sep="\t", header=0)
    print(Counter(siteData['y']))

    if enzyme is not None:
        generate_sample_specific_training_data(mod_file=out_file, header=True, db=out_db, flank_length=flank_length,
                                               enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                                               min_peptide_length=min_peptide_length,
                                               max_peptide_length=max_peptide_length,
                                               min_n = min_n,
                                               out_dir=out_dir)


def getTrainDataFromTable(db: str, site_table: str, ptm_type="sty", out_dir="./", flank_length=15, enzyme=None,
                          max_missed_cleavages=2,
                          min_peptide_length=7, max_peptide_length=45,
                          split_ratio=0.1, min_n=10, refine_file=None):
    """
    Prepare modification for general model training from a table and a protein fasta file
    :param db:
    :param ofile:
    :return:
    """

    generate_sample_specific_training_data(mod_file=site_table, ptm_type=ptm_type, header=True, db=db,
                                           flank_length=flank_length,
                                           enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                                           min_peptide_length=min_peptide_length,
                                           max_peptide_length=max_peptide_length,
                                           out_dir=out_dir,
                                           split_ratio=split_ratio,min_n=min_n, refine_file=refine_file)


def filter_by_similarity(input_file: str, similarity_threshold=0.5, out_dir="./"):
    siteData = pd.read_table(input_file, sep="\t", header=0)
    print("The number of modification sites in file %s : %d" % (input_file, siteData.shape[0]))

    out_file = out_dir + "/all_pep.fasta"
    fa_file = open(out_file, "w")

    for i, row in siteData.iterrows():
        pep = row['x']
        fa_file.write(">" + pep + "\n" + pep + "\n")

    fa_file.close()

    cd_hit_res = out_dir + "/cd_hit.txt"
    cd_hit = "cd-hit-2d -T 0 -d 0 -n 3 -M 0 -i %s -i2 %s -o %s -c %f" % (
        out_file, out_file, cd_hit_res, similarity_threshold)
    returned_value = subprocess.call(cd_hit, shell=True)
    print('returned value:', returned_value)


def filter_db(db=None, pattern=None, out_db=None):
    db_file = open(out_db, "w")
    n_total = 0
    n = 0
    for record in SeqIO.parse(db, "fasta"):
        n_total = n_total + 1
        proSeq = str(record.seq)
        if pattern in record.id:
            n = n + 1
            db_file.write(">" + record.id + "\n" + proSeq + "\n")

    print("Total sequences: %d, matched sequences: %d in file: %s." % (n_total, n, db))


def removeRedundantSite(input_data: str, outfile: str):
    """
    Remove redundant site records.
    :param file: The file generated by function getTrainDataFromPhosphoSitePlusFasta or getTrainDataFromFasta
    :param outfile:
    :return:
    """
    siteData = pd.read_table(input_data, sep="\t", header=0)

    ## If there are more than two records for the same peptide sequences, we only keep the modified one.
    siteData = siteData.sort_values(by=['x', 'y'], axis=0, ascending=[1, 0])
    siteData = siteData.drop_duplicates(subset=['aa', 'x'], keep='first')

    print(Counter(siteData['y']))
    ## save the new data to a file
    print("Save data to file:", outfile)
    siteData.to_csv(outfile, header=True, sep="\t", index=False)


def filterSites(input_data1: str, input_data2: str, outfile: str):
    """
    Remove records in input_data1, which are exists in input_data2 based on column flank.
    :param input_data1:
    :param input_data2:
    :param outfile:
    :return:
    """

    siteData1 = pd.read_table(input_data1, sep="\t", header=0)
    siteData2 = pd.read_table(input_data2, sep="\t", header=0)

    print("Rows in file %s is %d, rows in file %s is %d" % (
        input_data1, siteData1.shape[0], input_data2, siteData2.shape[0]))

    newData = siteData1[~np.in1d(siteData1['x'].tolist(), siteData2['x'].tolist())]

    print("Remove rows: %d" % (siteData1.shape[0] - newData.shape[0]))

    print("Save data to file:", outfile)
    newData.to_csv(outfile, header=True, sep="\t", index=False)


def importData4PeptideDetectabilityPrediction(file: str, db: str, enzyme="trypsin", max_missed_cleavages=2, outdir="./",
                                              balance=False):
    """
    Generate training data for peptide detectability prediction.
    :param file: A peptide file containing the identified peptides. There are two columns in this file. One is peptide
    and the other one is protein.
    :param db: A protein database file
    :param enzyme: Enzyme for protein digestion. Default is trypsin
    :param max_missed_cleavages: The maximum of missed cleavages
    :return: A pandas data frame
    """

    ## peptide\tprotein
    pep_data = pd.read_table(file, sep="\t", header=0)
    pepSet = set()
    proteinSet = set()
    min_peptide_length = 100
    max_peptide_length = 0
    for i, row in pep_data.iterrows():
        peptide = row['peptide']
        protein = row['protein']
        protein = str(protein)
        for acc in protein.split(";"):
            proteinSet.add(acc)

        pepSet.add(peptide)

        if len(peptide) > max_peptide_length:
            max_peptide_length = len(peptide)

        if len(peptide) < min_peptide_length:
            min_peptide_length = len(peptide)

    print("Identified unique peptides:", len(pepSet))
    print("Identified unique proteins:", len(proteinSet))

    print("Min peptide length:", min_peptide_length)
    print("Max peptide length:", max_peptide_length)

    ##
    neg_pepSet = set()
    print("Digest proteins in file:", db)
    print("Enzyme:", enzyme, ",", "max_missed_cleavages:", max_missed_cleavages, ",", "min_length:", min_peptide_length)
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proSeq = proSeq.replace('#', '')
        if record.id in proteinSet:
            # A set of unique (!) peptides.
            pSet = parser.cleave(proSeq, parser.expasy_rules[enzyme], max_missed_cleavages,
                                 min_length=min_peptide_length)
            for pep in pSet:
                if (len(pep) <= max_peptide_length) and (pep not in pepSet) and (len(pep) >= min_peptide_length):
                    neg_pepSet.add(pep)

    print("Negtive peptides:", len(neg_pepSet))

    ##
    if balance is True:
        used_neg_pepSet = random.sample(neg_pepSet, len(pepSet))
    else:
        used_neg_pepSet = random.sample(neg_pepSet, len(neg_pepSet))

    df1 = pd.DataFrame(list(pepSet), columns=['x'])

    pos_file = outdir + "/positive_peptide.txt"
    df1.to_csv(pos_file, sep="\t", index=False)
    print("Export positive peptides to file:", pos_file)

    df1['y'] = 1

    df2 = pd.DataFrame(list(used_neg_pepSet), columns=['x'])
    neg_file = outdir + "/negative_peptide.txt"
    df2.to_csv(neg_file, sep="\t", index=False)
    print("Export negative peptides to file:", neg_file)
    df2['y'] = 0

    df_all = df1.append(df2)
    all_file = outdir + "/positive_negative_peptides.txt"

    df_all.to_csv(all_file, sep="\t", index=False)
    return df_all


def get_type_of_input_ptm_site_db(db: str, ptm_type="sty"):
    """
    This function is used to determine the input data file type for preparing training or testing data
    :param db:
    :param ptm_type:
    :return:
    """
    input_type = -1  # not valid type
    # ptmMap = {'s': 1, 't': 1, 'y': 1}

    if db.lower().endswith(".fasta") or db.lower().endswith(".fa") or db.lower().endswith(".fas"):
        # use lowercase
        ptmMap = {}
        for aa in ptm_type:
            ptmMap[aa.lower()] = 1

        for record in SeqIO.parse(db, "fasta"):
            # print(record.id+"\t"+str(record.seq))
            proSeq = str(record.seq)
            if "#" in proSeq:
                # There is "#" after the modification site
                input_type = 1
                break
            else:
                for aa in ptmMap.keys():
                    # "s" in proSeq or "t" in proSeq or "y" in proSeq:
                    # The modification site is lowercase
                    if aa in proSeq:
                        input_type = 2
                        break

            if input_type != -1:
                break
    else:
        # a txt table: the first column is protein ID, the second column is modification position (1-based)
        input_type = 3

    print("Input type is %d ." % (input_type))
    return input_type


def process_mutation_data(input_file: str, db, ptm_type="sty", window_size=7, flank_length=15, out_dir="./"):
    proMap = dict()
    for record in SeqIO.parse(db, "fasta"):
        # print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proMap[record.id] = proSeq

    # 09CO014-varInfo.txt
    input_data = pd.read_table(input_file, sep="\t", header=0)

    # ptmMap = {'S': 1, 'T': 1, 'Y': 1}
    ptmMap = {}
    for aa in ptm_type:
        ptmMap[aa.upper()] = 1

    out_file = out_dir + "/variant_data.txt"
    o_file = open(out_file, 'w')
    o_file.write("%s\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))

    n_not_valid_rows = 0
    n_non_point_mutations = 0
    n_mMut_rows = 0
    n_non_mMut_rows = 0
    for i, row in input_data.iterrows():

        aa_ref = row['AA_Ref']
        aa_pos = row['AA_Pos']
        aa_var = row['AA_Var']

        if aa_ref is np.nan or aa_var is np.nan:
            n_not_valid_rows = n_not_valid_rows + 1
            continue

        # print(aa_ref)
        # print(aa_var+"\t"+aa_ref+"\t"+str(aa_pos))
        match_aa_ref = re.match(r'[A-Z]', aa_ref)
        match_aa_var = re.match(r'[A-Z]', aa_var)

        if match_aa_ref and match_aa_var and len(aa_ref) == 1 and len(aa_var) == 1:

            # only for point mutation
            # This is the point mutation position in a protein
            aa_pos = int(aa_pos)

            if "Protein" in input_data.columns.values:
                protein_ID = row['Protein']
            else:
                protein_ID = ""

            if "Variant_ID" in input_data.columns.values:
                vid = row['Variant_ID']
            else:
                vid = ""

            if (vid not in proMap.keys()):
                if protein_ID not in proMap:
                    print("Ignore: sequence is not found for %s: %s" % (protein_ID, "\t".join([str(x) for x in row])))
                    continue

                seq_mut = proMap[protein_ID]
                if aa_pos > len(seq_mut):
                    print("Ignore: length problem for %s: %s" % (protein_ID, "\t".join([str(x) for x in row])))
                    continue
                else:
                    seq_mut = list(seq_mut)
                    seq_mut[aa_pos - 1] = aa_var
                    seq_mut = "".join(seq_mut)
            else:
                seq_mut = proMap[vid]

            if protein_ID not in proMap.keys():
                seq_ref = proMap[vid]
                seq_ref = list(seq_ref)
                seq_ref[aa_pos - 1] = aa_ref
                seq_ref = "".join(seq_ref)
            else:
                seq_ref = proMap[protein_ID]

            # print("ok")

            # start position in protein
            start_pos = aa_pos - window_size
            if start_pos < 1:
                start_pos = 1

            # end position in protein
            end_pos = aa_pos + window_size
            if end_pos > len(seq_ref):
                end_pos = len(seq_ref)

            n_mMut = 0
            for pos in range(start_pos, end_pos + 1):
                aa = seq_ref[pos - 1]
                aa2 = seq_mut[pos - 1]
                diff_pos = aa_pos - pos
                # print(str(pos)+"\t"+aa+"\t"+aa2)
                if aa in ptmMap.keys():
                    flankSeq_ref = getPeptideSequence(seq_ref, pos, length=flank_length)
                    flankSeq_mut = getPeptideSequence(seq_mut, pos, length=flank_length)
                    o_file.write("%s\t%d\t%d\t%s\t%s\n" % (
                        '\t'.join(str(v) for v in row), pos, diff_pos, flankSeq_ref, flankSeq_mut))
                    n_mMut = n_mMut + 1
                    # print(vid+"\t"+str(aa_pos)+"\t"+str(pos)+"\t"+aa)
                elif aa2 in ptmMap.keys():
                    flankSeq_ref = getPeptideSequence(seq_ref, pos, length=flank_length)
                    flankSeq_mut = getPeptideSequence(seq_mut, pos, length=flank_length)
                    o_file.write("%s\t%d\t%d\t%s\t%s\n" % (
                        '\t'.join(str(v) for v in row), pos, diff_pos, flankSeq_ref, flankSeq_mut))
                    n_mMut = n_mMut + 1
                    # print(vid + "\t" + str(aa_pos) + "\t" + str(pos) + "\t" + aa2)

            if n_mMut >= 1:
                n_mMut_rows = n_mMut_rows + 1
            else:
                n_non_mMut_rows = n_non_mMut_rows + 1
        else:
            n_non_point_mutations = n_non_point_mutations + 1

    o_file.close()
    print("Not valid rows:", n_not_valid_rows)
    print("Non point mutations:", n_non_point_mutations)
    print("Valid mutations:", n_mMut_rows)
    print("Non valid mutations:", n_non_mMut_rows)
    return out_file


def variant_annotation_vcf(input_file: str, paras_file: str, out_dir: str):
    '''
    Perform variant annotation using annovar
    :param file: A single VCF file
    :param paras_file: A json format parameter file
    :param out_dir: Output directory
    :return:
    '''

    ## step 1: variant annotation based on annovar
    with open(paras_file, "r") as read_file:
        paras = json.load(read_file)

    ## path of annovar
    if "annovar_path" not in paras:
        print("Annovar path is not found!\n")
        os._exit(0)

    if "perl" in paras:
        vcf_cmd = paras['perl']
    else:
        vcf_cmd = "perl "

    vcf_cmd = vcf_cmd + paras['annovar_path'] + "/table_annovar.pl"
    vcf_cmd = vcf_cmd + " " + input_file

    if "database_path" not in paras:
        print("The database path for annovar is not found!\n")
        os.execl(0)

    vcf_cmd = vcf_cmd + " " + paras['database_path']

    genome_version = "hg38"
    if "genome_version" in paras:
        genome_version = paras['--buildver']
    else:
        print("Use hg38!\n")

    for op in paras:
        if op.startswith("-"):
            if op == "--thread":
                if paras[op] == 0:
                    vcf_cmd = vcf_cmd + " " + op + " " + str(mp.cpu_count())
                else:
                    vcf_cmd = vcf_cmd + " " + op + " " + str(paras[op])
            elif op == "--maxgenethread":
                if paras[op] == 0:
                    vcf_cmd = vcf_cmd + " " + op + " " + str(mp.cpu_count())
                else:
                    vcf_cmd = vcf_cmd + " " + op + " " + str(paras[op])
            else:
                vcf_cmd = vcf_cmd + " " + op + " " + str(paras[op])

    ## get parameters from json file
    if "--nastring" not in paras:
        vcf_cmd = vcf_cmd + " --nastring . "

    if input_file.endswith(".vcf") or input_file.endswith(".VCF"):
        if "--vcfinput" not in paras:
            vcf_cmd = vcf_cmd + " --vcfinput "

    if "--thread" not in paras:
        vcf_cmd = vcf_cmd + " --thread " + str(mp.cpu_count())

    if "--maxgenethread" not in paras:
        vcf_cmd = vcf_cmd + " --maxgenethread " + str(mp.cpu_count())

    if "--polish" not in paras:
        vcf_cmd = vcf_cmd + " --polish "

    input_file_name = os.path.basename(input_file)
    input_file_name = re.sub(".vcf$", "", input_file_name) + "_anno"
    vcf_cmd = vcf_cmd + " --outfile " + out_dir + "/" + input_file_name

    print("run => %s" % (vcf_cmd))
    os.system(vcf_cmd)

    ## build customized database

    multianno_txt = out_dir + "/" + input_file_name + "." + genome_version + "_multianno.txt"
    seq_fa = out_dir + "/" + input_file_name + ".refGene.fa"

    res = prepare_data_from_annovar(multianno_txt, seq_fa, out_dir)
    return res


def prepare_data_from_annovar(input_file: str, db: str, out_dir: str):
    '''
    Only consider nonsynonymous SNV
    :param input_file:
    :param db:
    :param out_file:
    :return:
    '''

    out_txt = out_dir + "/" + "out.txt"
    of = open(out_txt, "w")

    with open(input_file, "r") as f:
        line_number = 0
        hmap = dict()
        head_list = list()
        for line in f:
            line = line.strip()
            d = line.split("\t")
            if line_number == 0:
                ## head line
                head_list = d
                for i in range(0, len(d)):
                    hmap[d[i]] = i
                d[hmap['AAChange.refGene']] = "mRNA\tProtein\tAA_Ref\tAA_Pos\tAA_Var"
                of.write("\t".join(d[0:hmap['Otherinfo']]) + "\n")
            else:
                mutation_type = d[hmap['ExonicFunc.refGene']]
                if mutation_type == "nonsynonymous SNV":
                    aa_change = d[hmap['AAChange.refGene']]
                    mrnas = aa_change.split(",")
                    for mrna in mrnas:
                        res = mrna.split(":")
                        gene = res[0]
                        mrna_id = res[1]
                        aa = res[4]
                        rem = re.match(r"^p.([A-Z])(\d+)([A-Z])$", aa)
                        if rem is not None:
                            d[hmap['AAChange.refGene']] = mrna_id + "\t" + mrna_id + "\t" + "\t".join(rem.groups())
                            d[hmap['Gene.refGene']] = gene
                            of.write("\t".join(d[0:hmap['Otherinfo']]) + "\n")
            line_number = line_number + 1

    of.close()

    out_db = out_dir + "/" + "out.fa"
    od = open(out_db, "w")
    seqID = dict()
    for record in SeqIO.parse(db, "fasta"):
        if "WILDTYPE" in record.description:
            d = record.description.split(" ")
            id = d[1]

            if id not in seqID:
                seq = str(record.seq)
                seq = re.sub(r"\*$", "", seq)
                od.write(">" + id + "\n" + seq + "\n")
                seqID[id] = seq
    od.close()

    return [out_txt, out_db]

