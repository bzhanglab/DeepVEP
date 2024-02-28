import shap
import json
import numpy as np
import pandas as pd
import os
from .DataIO import processing_prediction_data, getAllModificationSites
from .Utils import combine_rts
from .Metrics import evaluate_model, add_confidence_metrics
import gc
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

#tf.compat.v1.disable_v2_behavior()

def get_default_bg_file(model_dir:str):
    bg_file = model_dir + "/bg_file.tsv"
    bg_db = model_dir + "/bg.fasta"
    if os.path.isfile(bg_file) and os.path.isfile(bg_db):
        print("Use default background file for model interpretability: %s" % (bg_file))
        return [bg_file,bg_db]
    else:
        print("There is no background file found for model interpretability in folder %s" % (bg_file))
        return None

def model_explainer(model, x, bg_data, out_dir="./", prefix="test"):

    # model is either a keras model or a keras model file
    if isinstance(model,str):
        model = tf.keras.models.load_model(model)

    explainer = shap.GradientExplainer(model, bg_data)
    if x.shape[0] == 1 and len(x.shape) <= 2:
        x = np.array([x])

    s_values = explainer.shap_values(x)
    s_raw_file = out_dir + "/shap_raw_" + str(prefix) + ".pkl"
    with open(s_raw_file, 'wb') as f:
        pickle.dump(s_values, f, pickle.HIGHEST_PROTOCOL)

    s_v = np.sum(s_values[0], axis=2)
    s_v_df = pd.DataFrame(s_v)
    s_file = out_dir + "/model_" + str(prefix) + "_shap.tsv"
    s_v_df.to_csv(s_file,sep="\t",header=False,index=False)


def generate_bg_samples(bg_file:str,n_each_class=500,out_dir="./",random_state=2021):
    a = pd.read_csv(bg_file,sep="\t",low_memory=False)
    min_n = min(min(a['y'].value_counts()),n_each_class)
    data_use = a.groupby('y',group_keys=False).apply(lambda x: x.sample(min_n,replace=False,random_state=random_state,axis=0))
    out_file = out_dir+"/bg_file.tsv"
    data_use.to_csv(out_file,sep="\t",index=False)
    return out_file


def model_explainer_test(model_file:str, test_file:str, bg_file:str, db=None, out_dir="./", prefix="test", method = "average",add_confidence_metric="fpr"):

    ## detect ptm type in the model file
    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)

    out_file = "-"

    if test_file is None:
        ## extract sites from database file
        print("Predict all sites in database file %s" % (db))
        test_file = out_dir + "/all_sites.tsv"
        getAllModificationSites(db, out_file=test_file, ptm_type=model_list['aa'])
    else:
        print("Predict sites in file %s" % (test_file))
    res_to_file = np.empty(0)
    ## prediction result
    y_pr_final = np.empty(0)

    if method == "average":
        print("Average ...")

        input_data = pd.read_table(test_file, sep="\t", header=0, low_memory=False)

        res = dl_models_predict(model_file, test_file=test_file, bg_file=bg_file, db=db, out_dir=out_dir, prefix=prefix)
        rt_pred = np.apply_along_axis(combine_rts, 1, res, method="mean", remove_outlier=True)

        if "y" in input_data.columns.values:
            y_true = np.asarray(input_data['y'])

        #y_pr_final = res.mean(axis=1)
        #y_pr_final = np.median(res,axis=1)


        #y_pr_final = np.apply_along_axis(combine_rts, 1, res, reverse=False, method="mean", remove_outlier=True)
        #y_pr_final = np.apply_along_axis(combine_rts, 1, res, reverse=False, method="median", remove_outlier=False)
        #y_pr_final = np.apply_along_axis(combine_rts, 1, res, reverse=False, method="median", remove_outlier=False)

        #np.save("res", res)
        #np.save("y_pr_final", y_pr_final)
        res_to_file = res
        #res_to_file = np.append(res_to_file, y_pr_final.reshape([y_pr_final.shape[0], 1]), axis=1)



        # rt_pred = minMaxScoreRev(y_pr_final, model_list['min_rt'], model_list['max_rt'])


        input_data['y_pred'] = rt_pred

        if add_confidence_metric is not None:
            met_res = add_confidence_metrics(input_data['y_pred'],os.path.dirname(model_file),metric=add_confidence_metric)
            if met_res is not None:
                input_data[add_confidence_metric] = met_res

        ## output
        out_file = out_dir + "/" + prefix + ".tsv"
        input_data.to_csv(out_file,sep="\t",index=False)

        ## evaluate
        if "y" in input_data.columns.values:
            #y_true = minMaxScale(np.asarray(input_data['y']), model_list['min_rt'], model_list['max_rt'])
            #y_pr_final = minMaxScale(np.asarray(input_data['y']), model_list['min_rt'], model_list['max_rt'])
            out_prefix = prefix + "_" + "evaluate"
            evaluate_model(input_data['y'], rt_pred, out_dir=out_dir, prefix=out_prefix)

    return out_file


def dl_models_predict(model_file:str, test_file:str, bg_file:str, db:str, batch_size=2048, out_dir="./", prefix="test"):

    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)

    y_dp = np.zeros(0)

    model_folder = os.path.dirname(model_file)
    avg_models = list()
    for (name, dp_model) in model_list['dp_model'].items():
        print("\nDeep learning model:", name)
        # keras model evaluation: loss and accuracy
        # load model
        dp_model_file = dp_model['model']
        flank_length = int((dp_model['peptide_length'] - 1)/2)

        ## prepare prediction data
        x, siteData = processing_prediction_data(test_file, db=db, flank_length=flank_length)

        ## prepare background data for shap, the format of this file is similar to prediction file
        bg_x, bg_siteData = processing_prediction_data(bg_file, db=db, flank_length=flank_length)

        model_name = os.path.basename(dp_model_file)
        model_full_path = model_folder + "/" + model_name

        model = load_model(model_full_path)
        avg_models.append(model)
        print("Shape for x:")
        print(x.shape)
        print("Shape for model input:")
        print(model._layers[0]._batch_input_shape)

        if(batch_size > x.shape[0]):
            batch_size = 32
        y_prob = model.predict(x, batch_size=batch_size)
        ## for class 1
        #y_prob_dp_vector = y_prob[:, 1]
        y_prob_dp_vector = y_prob
        y_prob_dp = y_prob_dp_vector.reshape([y_prob_dp_vector.shape[0], 1])

        siteData['y_pred'] = y_prob_dp
        siteData.to_csv(out_dir + "/model_" + str(name) + "_pred.tsv",sep="\t",header=True,index=False)
        if y_dp.shape[0] != 0:
            y_dp = np.append(y_dp, y_prob_dp, axis=1)
        else:
            y_dp = y_prob_dp

        test_data = pd.read_table(test_file,sep="\t",header=0)
        if 'y' in test_data.columns:
            y = np.array(test_data['y'].values)
            evaluation_res = model.evaluate(x, y, batch_size=batch_size)
            print("Metrics:")
            print(evaluation_res)

            # ROC
            if len(y.shape) >= 2:
                y_true_class = np.argmax(y, axis=1)
            else:
                y_true_class= y
            out_prefix = prefix + "_" + str(name)
            evaluate_model(y_true_class, y_prob_dp_vector, out_dir=out_dir, prefix=out_prefix)

        explainer = shap.GradientExplainer(model, bg_x[np.random.choice(bg_x.shape[0], 1000, replace=False)])
        np.save("bg_x.npy",bg_x)
        np.save("x.npy", x)
        model.save("shap_model.h5")
        if x.shape[0]  == 1 and len(x.shape) <= 2:
            x = np.array([x])

        shap_values = explainer.shap_values(x)
        shap_file = out_dir + "/shap_" + str(name) + ".pkl"
        with open(shap_file, 'wb') as f:
            pickle.dump(shap_values, f, pickle.HIGHEST_PROTOCOL)

        gc.collect()
        K.clear_session()

    return y_dp



