
import pandas as pd
import numpy as np
import os
import json

from Bio import SeqIO
import re
import sys
from os import listdir
from os.path import isdir

from .DataIO import getPeptideSequence
from .PTModels import ptm_predict
from .Utils import add_ptm_column



def mutation_impact_prediction_for_multiple_ptms(model_dir=None, input_file=None, db=None, window_size=7,
                                                 prefix="deepmvp", out_dir="./", ensemble_method="average",
                                                 add_model_explain=False, bg_data=None):
    model_folders = [os.path.join(model_dir,f) for f in listdir(model_dir) if isdir(os.path.join(model_dir,f))]
    res_files = dict()
    for m_dir in model_folders:
        model_file = os.path.join(m_dir,"model.json")
        ptm_name = os.path.basename(m_dir)
        o_dir = os.path.join(out_dir,ptm_name)
        if isdir(o_dir) is False:
            os.makedirs(o_dir)
        print("Predict mutation impact on PTM: %s, output folder: %s \n" % (ptm_name, o_dir))
        out_file = mutation_impact_prediction(model_file=model_file,input_file=input_file,db=db,window_size=window_size,prefix=prefix,out_dir=o_dir,
                                              add_model_explain=add_model_explain, bg_data=bg_data)
        if os.path.isfile(out_file):
            res_files[ptm_name] = out_file
        else:
            print("No result for %s" % (ptm_name))

    res = pd.concat([add_ptm_column(f,ptm_name) for ptm_name, f in res_files.items()],axis=0)
    o_file = out_dir + "/" + str(prefix) + "-mutation_impact.tsv"
    res.to_csv(o_file, sep="\t", index=False)



def detect_variant_file_type(input_file:str):
    # Support two formats: 1, CustomProDBJ, 2, VEP txt output format
    op = open(input_file, 'r')
    line = op.readline().strip()
    op.close()
    d = line.split("\t")
    if ("Protein" in d) and ("AA_Ref" in d) and ("AA_Pos" in d) and ("AA_Var" in d):
        print("TSV format!")
        return "tsv"
    else:
        print("VEP format!")
        return "vep"

def get_header_index_for_vep(input_file):
    # Opening file
    op = open(input_file, 'r')
    count = 0
    for line in op:
        if line.startswith("##"):
            count = count + 1
        else:
            break
    op.close()
    return count

def mutation_impact_prediction(model_file=None, input_file=None, db=None, window_size=7, prefix="deepmvp", out_dir="./",
                               ensemble_method="average",add_model_explain=False, bg_data=None):

    ## The following columns must be present in the input_file:
    ## Protein, AA_Ref, AA_Pos, AA_Var, pos

    ## For single mutation only

    ## For multiple mutations
    ## AA_Ref, AA_Pos, AA_Var
    ## A;D 2;3 B;C. Separated by ";"

    with open(model_file, "r") as read_file:
        model_info = json.load(read_file)
    ptm_aa = "-"
    if 'aa' in model_info:
        ptm_aa = model_info['aa']
    else:
        print("Error: aa is not present in model file!")
        sys.exit(1)

    file_map = process_mutation_data(input_file, db=db, ptm_aa=ptm_aa, window_size=window_size,
                                     prefix=prefix, out_dir=out_dir)

    ## check file if it's empty
    p_site_data = pd.read_table(file_map['ptm_site'],sep="\t",header=0,low_memory=False)
    if len(p_site_data) == 0:
        print("No site needs to be predicted!\n")
        out_file = ""
    else:
        ptm_pred_file = ptm_predict(model_file=model_file, test_file=file_map['ptm_site'], db=file_map['db'],
                                    prefix=prefix + "-ptm_pred", out_dir=out_dir,
                                    add_model_explain=add_model_explain, bg_data=bg_data)

        print(file_map['mutation'])
        mut_data = pd.read_table(file_map['mutation'], sep="\t", header=0, low_memory=False)

        site_prob_data= pd.read_table(ptm_pred_file, sep="\t", header=0, low_memory=False)
        site_prob_data = site_prob_data[["protein","pos","y_pred"]]

        ## reference peptide
        m_data = pd.merge(mut_data, site_prob_data.rename(columns={"protein": "Protein", "y_pred": "w_prob"}),
                          on=("Protein", "pos"), how="left")

        ## variant peptide
        m_data = pd.merge(m_data, site_prob_data.rename(columns={"protein": "Variant_ID", "y_pred": "m_prob"}),
                          on=("Variant_ID", "pos"), how="left")

        m_data['w_prob'].fillna(0, inplace=True)
        m_data['m_prob'].fillna(0, inplace=True)

        m_data['delta_prob'] = m_data['m_prob'] - m_data['w_prob']

        ## Joint probability
        # 1 or 'columns': apply function to each row.
        #m_data.to_csv("test.tsv", sep="\t", index=False)
        m_data['joint_prob'] = m_data.apply(calc_joint_probability, axis=1)

        ## Relative log fold changes of odds
        m_data['log_ratio'] = m_data.apply(calc_log_fold_changes_of_odds, axis=1)

        if add_model_explain:
          m_data = calc_evalue(m_data,model_dir=os.path.dirname(model_file))

        out_file = out_dir + "/" + str(prefix) + "-mutation_impact.tsv"

        #input_data.to_csv(out_file, sep="\t", index=False)
        m_data.to_csv(out_file, sep="\t", index=False)
    return out_file

def calc_evalue(a,model_dir):
    ## calculate evalue
    ptm_name = os.path.basename(model_dir)
    bg_score_increase_file = model_dir + "/" + ptm_name + "_increase.tsv"
    bg_score_decrease_file = model_dir + "/" + ptm_name + "_decrease.tsv"

    bg_score_file = model_dir + "/" + ptm_name + ".tsv"

    if os.path.isfile(bg_score_decrease_file) and os.path.isfile(bg_score_decrease_file):
        print("Use background score file %s and %s for evalue calculation." % (
        bg_score_increase_file, bg_score_decrease_file))
        bg_score1 = pd.read_csv(bg_score_increase_file, sep="\t", low_memory=False)
        bg_score2 = pd.read_csv(bg_score_decrease_file, sep="\t", low_memory=False)
        bg_score1 = np.array(bg_score1['delta_prob'])
        bg_score2 = np.array(bg_score2['delta_prob'])
        bg_score2 = np.abs(bg_score2)
        a['evalue'] = 1
        for i, row in a.iterrows():
            if row['delta_prob'] > 0:
                a.loc[i, "evalue"] = (np.sum(bg_score1 > row['delta_prob']) + 1)/(bg_score1.shape[0])
            else:
                a.loc[i, "evalue"] = (np.sum(bg_score2 > np.abs(row['delta_prob'])) + 1) / (bg_score2.shape[0])

        return a
    elif os.path.isfile(bg_score_file):
        print("Use background score file %s for evalue calculation." % (bg_score_file))
        bg_score = pd.read_csv(bg_score_file, sep="\t", low_memory=False)
        bg_score = np.array(bg_score['delta_prob'])
        bg_score = np.abs(bg_score)
        a['evalue'] = 1
        for i, row in a.iterrows():
            a.loc[i, "evalue"] = (np.sum( bg_score > np.abs(row['delta_prob']) ) + 1) / (bg_score.shape[0])
        return a


def calc_joint_probability(x):
    p = 0
    if x['w_prob'] > x['m_prob']:
        ## loss
        p = x['w_prob'] * (1 - x['m_prob'])
    else:
        ## gain
        p = (1 - x['w_prob']) * x['m_prob']
    return p

def calc_log_fold_changes_of_odds(x):
    p_ref = x['w_prob']
    p_mut = x['m_prob']
    if p_ref == 0:
        p_ref = 1e-6
    if p_mut == 0:
        p_mut = 1e-6

    y = np.log2(p_ref/(1-p_ref)) - np.log2(p_mut/(1-p_mut))
    return y


def process_mutation_data_will_be_removed(input_file:str, db, ptm_aa="sty", window_size=7, prefix = "deepmvp", out_dir="./"):
    ## The following columns must be present in the input_file:
    ## Protein, AA_Ref, AA_Pos, AA_Var
    ## pos is 1-based
    proMap = dict()
    for record in SeqIO.parse(db, "fasta"):
        #print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proMap[record.id] = proSeq


    ## output files
    ## For PTM prediction
    site_file = out_dir + "/" + str(prefix) + "-site.tsv"
    site_db = out_dir + "/" + str(prefix) + "-site.fasta"

    ## For mutation impact
    ## pos     diff_pos        w_pep   m_pep   w_prob  m_prob  delta
    mutation_file = out_dir + "/" + str(prefix) + "-mutation.tsv"

    of_site = open(site_file,"w")
    of_db = open(site_db,"w")
    of_m = open(mutation_file,"w")

    save_pro_map = {}

    of_site.write("protein\taa\tpos\n")

    variant_file_type = detect_variant_file_type(input_file)
    if variant_file_type == "tsv":
        input_data = pd.read_table(input_file, sep="\t", header=0, low_memory=False)
    elif variant_file_type == "vep":
        head_index = get_header_index_for_vep(input_file)
        input_data = pd.read_table(input_file, skiprows=head_index, sep="\t", low_memory=False)
    else:
        print("Format is not supported!")
        sys.exit(1)

    # ptmMap = {'S': 1, 'T': 1, 'Y': 1}
    ptmMap = {}
    for aa in ptm_aa:
        ptmMap[aa.upper()] = 1

    if variant_file_type == "tsv":
        # CustomProDBJ
        if "Variant_ID" in input_data.columns.values:
            of_m.write("%s\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))
        else:
            of_m.write("%s\tVariant_ID\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))
    elif variant_file_type == "vep":
        # VEP
        of_m.write("%s\tVariant_ID\tProtein\tAA_Ref\tAA_Pos\tAA_Var\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))


    n_not_valid_rows = 0
    n_non_point_mutations = 0
    n_mMut_rows = 0
    n_non_mMut_rows = 0
    re_int_match = re.compile(r"^\d+$")
    for i, row in input_data.iterrows():
        ###################################################
        ## get protein ID and amino acid change information
        if variant_file_type == "tsv":
            aa_ref = row['AA_Ref']
            aa_pos = row['AA_Pos']
            aa_var = row['AA_Var']
            protein_ID = row['Protein']

            if "Variant_ID" in input_data.columns.values:
                vid = row['Variant_ID']
            else:
                vid = "var_"+str(i)
        elif variant_file_type == "vep":
            protein_ID = row['ENSP']
            aa_pos = row["Protein_position"]
            if re_int_match.search(str(aa_pos)):
                aa_pos = int(aa_pos)
                aa_ch = row['Amino_acids'].split("/")
                if len(aa_ch) < 2:
                    n_not_valid_rows = n_not_valid_rows + 1
                    continue
                aa_ref = aa_ch[0]
                aa_var = aa_ch[1]
            else:
                n_not_valid_rows = n_not_valid_rows + 1
                continue

            if aa_var == aa_ref:
                n_not_valid_rows = n_not_valid_rows + 1
                continue

            vid = "var_" + str(i)

        else:
            print("The file format is not supported: %s" % (input_file))
            sys.exit(1)

        ## get protein ID and amino acid change information. END
        ###################################################

        if aa_ref is np.nan or aa_var is np.nan:
            n_not_valid_rows = n_not_valid_rows + 1
            continue

        #print(aa_ref)
        #print(aa_var+"\t"+aa_ref+"\t"+str(aa_pos))
        match_aa_ref = re.match(r'[A-Z]', aa_ref)
        match_aa_var = re.match(r'[A-Z]', aa_var)

        if match_aa_ref and match_aa_var and len(aa_ref) ==1 and len(aa_var) ==1:

            # only for point mutation
            # This is the point mutation position in a protein
            aa_pos = int(aa_pos)
            seq_ref = "-"
            seq_mut = "-"

            if protein_ID not in proMap.keys():
                print("Protein sequence not found: %s" % (protein_ID))
                print("Ignore: " + "\t".join(map(str, row)))
                continue
                #sys.exit(1)
            else:
                seq_ref = proMap[protein_ID]
                if aa_pos > len(seq_ref):
                    print("Error: length problem for %s: %s" % (protein_ID, "\t".join([str(x) for x in row])))
                    print("Ignore: " + "\t".join(map(str, row)))
                    continue
                    #sys.exit(1)
                else:
                    if seq_ref[aa_pos - 1] != aa_ref:
                        print("\t".join(map(str, row)))
                        print("Ignore: the amino acid %s at %d in protein sequence doesn't match the one (%s) from input!" % (seq_ref[aa_pos - 1],aa_pos, aa_ref))
                        continue
                        #sys.exit(1)
                    else:
                        seq_mut = list(proMap[protein_ID])
                        seq_mut[int(aa_pos - 1)] = aa_var
                        seq_mut = "".join(seq_mut)

            seq_ref_list = list(seq_ref)
            seq_mut_list = list(seq_mut)

            # start position in protein
            start_pos = aa_pos - window_size
            if start_pos < 1:
                start_pos = 1

            # end position in protein
            end_pos = aa_pos + window_size
            if end_pos > len(seq_ref_list):
                end_pos = len(seq_ref_list)


            n_mMut = 0
            for pos in range(start_pos,end_pos+1):
                aa_in_ref = seq_ref_list[pos-1]
                aa_in_mut = seq_mut_list[pos-1]
                diff_pos = aa_pos - pos

                ## reference protein
                if aa_in_ref in ptmMap.keys() or aa_in_mut in ptmMap.keys():

                    if aa_in_ref in ptmMap.keys():
                        of_site.write(protein_ID+"\t"+aa_in_ref+"\t"+str(pos)+"\n")

                    if aa_in_mut in ptmMap.keys():
                        of_site.write(vid + "\t" + aa_in_mut + "\t" + str(pos) + "\n")

                    flankSeq_ref = getPeptideSequence(seq_ref,pos,window_size)
                    flankSeq_mut = getPeptideSequence(seq_mut,pos,window_size)

                    if variant_file_type == "tsv":
                        ## CustomProDBJ
                        if "Variant_ID" in input_data.columns.values:
                            of_m.write("%s\t%d\t%d\t%s\t%s\n" % ('\t'.join(str(v) for v in row), pos, diff_pos, flankSeq_ref, flankSeq_mut))
                        else:
                            of_m.write("%s\t%s\t%d\t%d\t%s\t%s\n" % ('\t'.join(str(v) for v in row), vid, pos, diff_pos, flankSeq_ref, flankSeq_mut))

                    elif variant_file_type == "vep":
                        ## VEP
                        of_m.write("%s\t%s\t%s\t%s\t%d\t%s\t%d\t%d\t%s\t%s\n" % ('\t'.join(str(v) for v in row), vid, protein_ID, aa_ref, aa_pos, aa_var, pos, diff_pos, flankSeq_ref, flankSeq_mut))


                    #o_file.write("%s\t%d\t%d\t%s\t%s\n" % ('\t'.join(str(v) for v in row), pos, diff_pos, flankSeq_ref, flankSeq_mut))
                    n_mMut = n_mMut + 1
                    of_db.write(">" + str(vid) + "\n" + seq_mut + "\n")
                    if protein_ID not in save_pro_map.keys():
                        of_db.write(">" + str(protein_ID) + "\n" + seq_ref + "\n")
                        save_pro_map[protein_ID] = 1

            if n_mMut >= 1:
                n_mMut_rows = n_mMut_rows + 1
            else:
                n_non_mMut_rows = n_non_mMut_rows + 1
        else:
            n_non_point_mutations = n_non_point_mutations + 1

    of_site.close()
    of_db.close()
    of_m.close()
    print("Not valid rows:", n_not_valid_rows)
    print("Non point mutations:",n_non_point_mutations)
    print("Valid mutations:", n_mMut_rows)
    print("Non valid mutations:", n_non_mMut_rows)

    site_data = pd.read_table(site_file,sep="\t")
    site_data.drop_duplicates(keep='first', inplace=True)
    site_data.to_csv(site_file,sep="\t", index=False)

    out_file_map = {}
    out_file_map['ptm_site'] = site_file
    out_file_map['db'] = site_db
    out_file_map['mutation'] = mutation_file
    return out_file_map

def process_mutation_data(input_file:str, db, ptm_aa="sty", window_size=7, prefix = "deepmvp", out_dir="./"):
    ## The following columns must be present in the input_file:
    ## Protein, AA_Ref, AA_Pos, AA_Var
    ## pos is 1-based
    proMap = dict()
    for record in SeqIO.parse(db, "fasta"):
        #print(record.id+"\t"+str(record.seq))
        proSeq = str(record.seq)
        proMap[record.id] = proSeq


    ## output files
    ## For PTM prediction
    site_file = out_dir + "/" + str(prefix) + "-site.tsv"
    site_db = out_dir + "/" + str(prefix) + "-site.fasta"

    ## For mutation impact
    ## pos     diff_pos        w_pep   m_pep   w_prob  m_prob  delta
    mutation_file = out_dir + "/" + str(prefix) + "-mutation.tsv"

    of_site = open(site_file,"w")
    of_db = open(site_db,"w")
    of_m = open(mutation_file,"w")

    save_pro_map = {}

    of_site.write("protein\taa\tpos\n")

    variant_file_type = detect_variant_file_type(input_file)
    if variant_file_type == "tsv":
        input_data = pd.read_table(input_file, sep="\t", header=0, low_memory=False)
    elif variant_file_type == "vep":
        head_index = get_header_index_for_vep(input_file)
        input_data = pd.read_table(input_file, skiprows=head_index, sep="\t", low_memory=False)
    else:
        print("Format is not supported!")
        sys.exit(1)

    # ptmMap = {'S': 1, 'T': 1, 'Y': 1}
    ptmMap = {}
    for aa in ptm_aa:
        ptmMap[aa.upper()] = 1

    if variant_file_type == "tsv":
        # CustomProDBJ
        if "Variant_ID" in input_data.columns.values:
            of_m.write("%s\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))
        else:
            of_m.write("%s\tVariant_ID\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))
    elif variant_file_type == "vep":
        # VEP
        of_m.write("%s\tVariant_ID\tProtein\tAA_Ref\tAA_Pos\tAA_Var\tpos\tdiff_pos\tw_pep\tm_pep\n" % ("\t".join(input_data.columns)))


    n_not_valid_rows = 0
    n_non_point_mutations = 0
    n_mMut_rows = 0
    n_non_mMut_rows = 0
    re_int_match = re.compile(r"^\d+$")
    for i, row in input_data.iterrows():
        ###################################################
        ## get protein ID and amino acid change information
        if variant_file_type == "tsv":
            aa_ref = row['AA_Ref']
            aa_pos = row['AA_Pos']
            aa_var = row['AA_Var']
            protein_ID = row['Protein']

            if "Variant_ID" in input_data.columns.values:
                vid = row['Variant_ID']
            else:
                vid = "var_"+str(i)
        elif variant_file_type == "vep":
            protein_ID = row['ENSP']
            aa_pos = row["Protein_position"]
            if re_int_match.search(str(aa_pos)):
                aa_pos = int(aa_pos)
                aa_ch = row['Amino_acids'].split("/")
                if len(aa_ch) < 2:
                    n_not_valid_rows = n_not_valid_rows + 1
                    continue
                aa_ref = aa_ch[0]
                aa_var = aa_ch[1]
            else:
                n_not_valid_rows = n_not_valid_rows + 1
                continue

            if aa_var == aa_ref:
                n_not_valid_rows = n_not_valid_rows + 1
                continue

            vid = "var_" + str(i)

        else:
            print("The file format is not supported: %s" % (input_file))
            sys.exit(1)

        ## get protein ID and amino acid change information. END
        ###################################################

        if protein_ID not in proMap.keys():
            n_not_valid_rows = n_not_valid_rows + 1
            print("Protein sequence not found: %s" % (protein_ID))
            print("Ignore: " + "\t".join(map(str, row)))
            continue

        p_site = get_ptm_site_candidate(aa_ref,aa_pos,aa_var,protein_ID,proMap[protein_ID], ptm_aa=ptm_aa, window_size=window_size)
        #print("ok1")
        #print(p_site)

        if p_site is None:
            n_not_valid_rows = n_not_valid_rows + 1
            continue
        else:
            ## affected sites
            affected_sites = dict()
            p_site['var_pro_id'] = vid

            ## export data for PTM site prediction
            for pos in p_site['ref_site_map'].keys():
                of_site.write(protein_ID + "\t" + p_site['ref_site_map'][pos] + "\t" + str(pos) + "\n")
                affected_sites[pos] = 1

            for pos in p_site['var_site_map'].keys():
                of_site.write(p_site['var_pro_id'] + "\t" + p_site['var_site_map'][pos] + "\t" + str(pos) + "\n")
                if pos in affected_sites.keys():
                    affected_sites[pos] = affected_sites[pos] + 1
                else:
                    affected_sites[pos] = 1

            if len(affected_sites) >= 1:
                n_mMut_rows = n_mMut_rows + 1
                ## export data for protein sequences
                if str(p_site['var_pro_id']) not in save_pro_map.keys():
                    of_db.write(">" + str(p_site['var_pro_id']) + "\n" + p_site['var_pro_seq'] + "\n")
                    save_pro_map[str(p_site['var_pro_id'])] = 1

                if protein_ID not in save_pro_map.keys():
                    of_db.write(">" + str(protein_ID) + "\n" + proMap[protein_ID] + "\n")
                    save_pro_map[protein_ID] = 1

                ## export data for variant information
                for pos in affected_sites.keys():
                    flankSeq_ref = getPeptideSequence(proMap[protein_ID], pos, window_size)
                    flankSeq_mut = getPeptideSequence(p_site['var_pro_seq'], pos, window_size)

                    diff_pos = get_distance(pos,aa_pos)
                    if variant_file_type == "tsv":
                        ## CustomProDBJ
                        if "Variant_ID" in input_data.columns.values:
                            of_m.write("%s\t%d\t%d\t%s\t%s\n" % (
                            '\t'.join(str(v) for v in row), pos, diff_pos, flankSeq_ref, flankSeq_mut))
                        else:
                            of_m.write("%s\t%s\t%d\t%d\t%s\t%s\n" % (
                            '\t'.join(str(v) for v in row), vid, pos, diff_pos, flankSeq_ref, flankSeq_mut))

                    elif variant_file_type == "vep":
                        ## VEP
                        of_m.write("%s\t%s\t%s\t%s\t%d\t%s\t%d\t%d\t%s\t%s\n" % (
                        '\t'.join(str(v) for v in row), vid, protein_ID, aa_ref, aa_pos, aa_var, pos, diff_pos, flankSeq_ref,
                        flankSeq_mut))

    of_site.close()
    of_db.close()
    of_m.close()
    print("Not valid rows:", n_not_valid_rows)
    print("Non point mutations:",n_non_point_mutations)
    print("Valid mutations:", n_mMut_rows)
    print("Non valid mutations:", n_non_mMut_rows)

    site_data = pd.read_table(site_file,sep="\t")
    site_data.drop_duplicates(keep='first', inplace=True)
    site_data.to_csv(site_file,sep="\t", index=False)

    out_file_map = {}
    out_file_map['ptm_site'] = site_file
    out_file_map['db'] = site_db
    out_file_map['mutation'] = mutation_file
    return out_file_map


def get_distance(target_pos:int,mut_pos):
    if ";" in str(mut_pos):
        a = [int(i) - target_pos for i in mut_pos.split(";")]
        a = np.sort(a)
        b = a[np.argmin(np.abs(a))]
    else:
        b = int(mut_pos) - target_pos

    return b


def get_ptm_site_candidate(aa_ref,aa_pos,aa_var,protein_ID,protein_seq, ptm_aa="sty", window_size=7):
    ## process one row from mutation data file
    # This is the point mutation position in a protein
    # m_dat: dataframe, "aa_pos", "aa_ref", 'aa_var'
    m_dat = format_mutation(aa_ref, aa_pos, aa_var)

    if m_dat is None or m_dat.shape[0] == 0:
        return None
    else:

        if m_dat.shape[0] == 1:
            var_id, seq_mut = get_mut_pro_seq(m_dat, protein_ID, protein_seq)
            aa_pos = int(aa_pos)
            ref_site_map = get_ptm_sites(aa_pos, pro_seq=protein_seq, ptm_aa=ptm_aa, window_size=window_size)
            var_site_map = get_ptm_sites(aa_pos, pro_seq=seq_mut, ptm_aa=ptm_aa, window_size=window_size)

        else:
            var_id, seq_mut = get_mut_pro_seq(m_dat, protein_ID, protein_seq)
            ref_site_map = dict()
            var_site_map = dict()
            for pos in m_dat['aa_pos']:
                ref_site_map_tmp = get_ptm_sites(pos, pro_seq=protein_seq, ptm_aa=ptm_aa, window_size=window_size)
                var_site_map_tmp = get_ptm_sites(pos, pro_seq=seq_mut, ptm_aa=ptm_aa, window_size=window_size)
                ref_site_map.update(ref_site_map_tmp)
                var_site_map.update(var_site_map_tmp)

        res = dict()
        res['var_pro_id'] = var_id
        res['var_pro_seq'] = seq_mut
        res['ref_site_map'] = ref_site_map
        res['var_site_map'] = var_site_map
        return res

def get_ptm_sites(pos:int, pro_seq, ptm_aa="STY", window_size=7):

    out_map = dict()

    start_pos = pos - window_size
    if start_pos < 1:
        start_pos = 1

    # end position in protein
    end_pos = pos + window_size
    if end_pos > len(pro_seq):
        end_pos = len(pro_seq)

    for pos in range(start_pos, end_pos + 1):

        aa_in_ref = pro_seq[pos - 1]

        if aa_in_ref in ptm_aa:


            if aa_in_ref in ptm_aa:
                out_map[pos] = aa_in_ref

    return out_map


def get_mut_pro_seq(var_df, protein_ID:str, pro_seq:str):

    # This can handle one mutation and multiple mutation combination
    # var_df is a dataframe with format like below:
    #   aa_ref  aa_pos aa_var
    # 0      a      10      b
    # 2      b      13      f
    # 1      c      14      d
    mut_seq_list = list(pro_seq)

    var_list = list()
    for i, row in var_df.iterrows():

        if pro_seq[int(row['aa_pos'] - 1)] != row['aa_ref']:
            print("Error: the amino acid %s at %d in protein sequence doesn't match the one (%s) from input!" % (pro_seq[int(row['aa_pos'] - 1)], row['aa_pos'], row['aa_ref']))
            print("\t".join(map(str, row)))
            sys.exit(1)

        if row['aa_pos'] > len(pro_seq):
            print("Error: length problem for %s: %s" % (protein_ID, "\t".join([str(x) for x in row])))
            print("\t".join(map(str, row)))
            sys.exit(1)

        mut_seq_list[int(row['aa_pos'] - 1)] = row['aa_var']
        var_list.append("".join(map(str, row)))

    var_ID = str(protein_ID) + ":" + ":".join(var_list)

    var_seq = "".join(mut_seq_list)

    return (var_ID,var_seq)


def format_mutation(aa_ref,aa_pos,aa_var):

    # check if mutation is valid or not
    # convert format

    if aa_ref is np.nan or aa_var is np.nan:
        return None

    if ";" in aa_ref:
        ## multiple mutations should be separated by ";"
        aa_ref_list = aa_ref.split(";")
        aa_pos_list = [int(i) for i in str(aa_pos).split(";")]
        aa_var_list = aa_var.split(";")

        #print(aa_ref_list)
        #print(aa_pos_list)
        #print(aa_var_list)

        if not (len(aa_ref_list) == len(aa_pos_list)) and (len(aa_pos_list) == len(aa_var_list)):
            #print('error1')
            return None

        m_dat = pd.DataFrame(list(zip(aa_ref_list, aa_pos_list, aa_var_list)), columns=['aa_ref', 'aa_pos', 'aa_var'])
        m_dat = m_dat.sort_values(["aa_pos", "aa_ref", 'aa_var'], ascending=(True, True, True))

    else:
        ## single mutation
        m_dat = pd.DataFrame(list(zip([aa_ref], [int(aa_pos)], [aa_var])), columns=['aa_ref', 'aa_pos', 'aa_var'])


    for i, row in m_dat.iterrows():
        match_aa_ref = re.match(r'^[A-Z]$', row['aa_ref'])
        match_aa_var = re.match(r'^[A-Z]$', row['aa_var'])
        match_aa_pos = re.match(r'^[0-9]+$', str(row['aa_pos']))
        if not (match_aa_ref and match_aa_var and match_aa_pos and len(row["aa_ref"]) == 1 and len(row["aa_var"]) == 1):
            ## not valid
            #print('error2')
            return None

    return m_dat


