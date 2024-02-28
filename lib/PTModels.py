
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, Bidirectional, LSTM, \
    Embedding, MaxPooling1D, Average, Bidirectional, GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, load_model, clone_model, model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import os
import gc
import tensorflow.keras.backend as K
import tensorflow as tf

from .RegCallback import RegCallback
from .DataIO import data_processing, processing_prediction_data, data_processing_range, getAllModificationSites
from .Utils import combine_rts,add_ptm_column
from .Metrics import evaluate_model, add_confidence_metrics
from .ModelView import model_explainer,generate_bg_samples,get_default_bg_file

import pickle
import json
import numpy as np
from shutil import copyfile
import sys

def train_model(input_data: str, test_file=None, db=None, batch_size=64, nb_epoch=100, early_stop_patience=None, out_dir="./", prefix = "test",
                p_model=None,peptide_length=None,
                model=None, optimizer_name=None,add_ReduceLROnPlateau=False,
                use_all_data=True,
                use_external_test_data=True, add_eval_callback=True,
                add_tb_callback=False, tf_learning=False, n_patience=5, lr=None, random_seed=2018,
                seq_encode_method="one_hot"):

    res_map = dict()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Build deep learning model ...")


    #X_train, Y_train, X_test, Y_test = data_processing(train_file=input_data, test_file = test_file,
    #                                                   use_all_data=use_all_data,out_dir=out_dir)
    flank_length = int((peptide_length-1)/2)
    dat = data_processing_range(train_file=input_data, test_file=test_file, db=db, use_all_data=use_all_data,out_dir=out_dir,
                                flank_length_range=[flank_length], random_seed=random_seed,
                                seq_encode_method=seq_encode_method)
    X_train, Y_train, X_test, Y_test = dat[peptide_length]

    if model is None:
        print("Error: there is no model file provided!")
        sys.exit(1)
    else:
        print("Use input model ...")
        if tf_learning is False:
            # if this is not transfer learning, we need to check the input shape. For transfer learning. The input share
            # cannot be changed.
            model = clone_model(model)
            model = change_model_input_shape(model, X_test.shape[1:])

    if p_model is not None:
        transfer_layer = 5
        frozen = True
        # model_copy.set_weights(model.get_weights())
        base_model = load_model(p_model)
        print("Perform transfer learning ...")
        n_layers = len(base_model.layers)
        print("The number of layers: %d" % (n_layers))
        for l in range((n_layers - transfer_layer)):
            if l != 0:
                model.layers[l].set_weights(base_model.layers[l].get_weights())
                if frozen is True:
                    model.layers[l].trainable = False
                    print("layer (frozen:True): %d %s" % (l,model.layers[l].name))
                else:
                    print("layer (frozen:False): %d %s" % (l,model.layers[l].name))

    if model.optimizer is None:
        ## use default optimizer: Adam
        if optimizer_name is None:
            print("Use default optimizer:Adam")
            model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
        else:
            print("Use optimizer provided by user: %s" % (optimizer_name))
            model.compile(loss='binary_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

    else:
        if optimizer_name is None:
            print("Use optimizer from the model.")
            if tf_learning:
                #print("Adam(1e-5)")
                if (lr is not None) and (lr > 0) :
                    print("Use learning rate: %f" % (lr))
                    model.compile(loss='binary_crossentropy',
                                  ## In this case, we cannot change the learning rate.
                                  optimizer=tf.keras.optimizers.Adam(lr),
                                  metrics=['accuracy'])
                else:
                    model.compile(loss='binary_crossentropy',
                                  ## In this case, we cannot change the learning rate.
                                  optimizer=tf.keras.optimizers.Adam(),
                                  metrics=['accuracy'])
            else:
                model.compile(loss='binary_crossentropy',
                              ## In this case, we cannot change the learning rate.
                              optimizer=model.optimizer,
                              metrics=['accuracy'])
        else:
            print("Use optimizer provided by user: %s" % (optimizer_name))
            model.compile(loss='binary_crossentropy',
                               ## In this case, we cannot change the learning rate.
                               optimizer=optimizer_name,
                               metrics = ['accuracy'])

    print("optimizer: %s" % (type(model.optimizer)))

    model.summary()


    # Save model
    model_chk_path = out_dir + "/best_model.hdf5"
    mcp = ModelCheckpoint(model_chk_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False,
                          verbose=1, mode='max')

    all_callbacks = list()
    #all_callbacks.append(my_callbacks)
    all_callbacks.append(mcp)

    if add_ReduceLROnPlateau is True:
        print("Use ReduceLROnPlateau!")
        all_callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=n_patience, factor=0.5, verbose=1,min_lr=0.000001,min_delta=0))

    if early_stop_patience is not None:
        print("Use EarlyStopping: %d" % (early_stop_patience))
        all_callbacks.append(EarlyStopping(patience=early_stop_patience,verbose=1, monitor="val_accuracy"))

    if add_tb_callback:
        print("Use TensorBoard.")
        log_dir = out_dir + "/tb_log_dir"
        all_callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))

    ## monitor training information
    # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    #model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, Y_test), callbacks=[my_callbacks, mcp])

    if isinstance(X_train, list):
        n_iteration = len(X_train)
        for i in range(n_iteration):
            print("\nIteration: ", i)

            callbacks_i = list(all_callbacks)
            if add_eval_callback:
                my_callbacks = RegCallback(X_train[i], X_test, Y_train[i], Y_test)
                callbacks_i.append(my_callbacks)

            if i != 0:
                print("Load best model from previous iteration.")
                model = load_model(model_chk_path)

            if use_external_test_data is True:
                model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=nb_epoch,
                          validation_data=(X_test, Y_test),
                          # callbacks=[my_callbacks, mcp])
                          callbacks=callbacks_i)
            else:
                model.fit(X_train[i], Y_train[i], batch_size=batch_size, epochs=nb_epoch,
                          validation_split=0.1,
                          # callbacks=[my_callbacks, mcp])
                          callbacks=callbacks_i)
    else:

        callbacks_i = list(all_callbacks)
        if add_eval_callback:
            my_callbacks = RegCallback(X_train, X_test, Y_train, Y_test)
            callbacks_i.append(my_callbacks)

        if use_external_test_data is True:
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                      validation_data=(X_test, Y_test),
                      # callbacks=[my_callbacks, mcp])
                      callbacks=callbacks_i)
        else:
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                      validation_split=0.1,
                      # callbacks=[my_callbacks, mcp])
                      callbacks=callbacks_i)
    #
    model_best = load_model(model_chk_path)

    #x = pd.DataFrame({"y": y_true, "y_pred": y_pred_rev.reshape(y_pred_rev.shape[0])})
    #out_file = out_dir + "/" + prefix +".csv"
    #print("Prediction result: %s" % (out_file))
    #x.to_csv(out_file)

    res_map['model'] = model_best
    res_map['max_x_length'] = X_test[0].shape[1]
    return res_map


def ensemble_models(input_data: str, test_file=None,
                    models_file=None, db=None,
                    ga_file=None,
                    ensemble_method="average",
                    batch_size=64, nb_epoch=100, out_dir="./", prefix="test",
                    early_stop_patience=None,
                    add_ReduceLROnPlateau=False,
                    add_eval_callback=False,
                    add_tb_callback=False,
                    gpu=1, # don't use
                    n_patience=5,
                    lr=None,
                    seq_encode_method="one_hot"):

    # test data
    X_test = np.empty(1)
    Y_test = np.empty(1)

    y_pr = []
    score = []

    model_list = dict()

    siteData = pd.read_table(input_data, sep="\t", header=0)
    model_list['aa'] = "".join(siteData['aa'].unique())
    if len(model_list['aa']) >= 2:
        print("Modification on amino acids %s" % (model_list['aa']))
    else:
        print("Modification on amino acid %s" % (model_list['aa']))


    if ga_file is not None:
        tf_learning = False
        model_file_path = ga_file
    else:
        tf_learning = True
        model_file_path = models_file

    print("Read models from file %s" % (model_file_path))
    with open(model_file_path, "r") as read_file:
        ga_model_list = json.load(read_file)

    if tf_learning:
        print("Perform transfer learning ...")
        ga_model_list = ga_model_list['dp_model']
    else:
        print("Train models from scratch ...")

    model_folder = os.path.dirname(model_file_path)

    model_list['dp_model'] = dict()

    # For each model, train the model
    random_i = 0
    for i in ga_model_list.keys():
        random_i = random_i + 1
        name = str(i)
        print("Train model:", )

        m_file = model_folder + "/" + os.path.basename(ga_model_list[i]['model'])
        print("Model file: %s -> %s" % (name, m_file))

        if gpu >= 2:
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
            n_gpu = strategy.num_replicas_in_sync

            with strategy.scope():
                if tf_learning:
                    model = tf.keras.models.load_model(m_file)
                else:
                    with open(m_file, "r") as json_read:
                        model = tf.keras.models.model_from_json(json_read.read())

                if "optimizer_name" in ga_model_list[i]:
                    optimizer_name = ga_model_list[i]['optimizer_name']
                else:
                    optimizer_name = None

                peptide_length = ga_model_list[i]['peptide_length']

                res_map = train_model(input_data=input_data, test_file=test_file, batch_size=batch_size * n_gpu,
                                      nb_epoch=nb_epoch, early_stop_patience=early_stop_patience, db=db,
                                      out_dir=out_dir, prefix=str(name), model=model, peptide_length=peptide_length,
                                      add_eval_callback=add_eval_callback,
                                      optimizer_name=optimizer_name, add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                                      add_tb_callback=add_tb_callback, tf_learning=tf_learning, random_seed=random_i,
                                      n_patience=n_patience,lr=lr,seq_encode_method=seq_encode_method)
        else:
            # use one GPU
            if tf_learning:
                model = tf.keras.models.load_model(m_file)
            else:
                with open(m_file, "r") as json_read:
                    model = tf.keras.models.model_from_json(json_read.read())

            if "optimizer_name" in ga_model_list[i]:
                optimizer_name = ga_model_list[i]['optimizer_name']
            else:
                optimizer_name = None

            peptide_length = ga_model_list[i]['peptide_length']

            res_map = train_model(input_data=input_data, test_file=test_file, batch_size=batch_size,
                                  nb_epoch=nb_epoch, early_stop_patience=early_stop_patience, db=db,
                                  out_dir=out_dir, prefix=str(name), model=model, peptide_length=peptide_length,
                                  add_eval_callback=add_eval_callback,
                                  optimizer_name=optimizer_name, add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                                  add_tb_callback=add_tb_callback, tf_learning=tf_learning, random_seed=random_i,
                                  n_patience=n_patience,lr=lr,seq_encode_method=seq_encode_method)

        ## save the model to a file:
        model_file_name = "model_" + str(name) + ".h5"
        model_file_path = out_dir + "/" + model_file_name
        res_map["model"].save(model_file_path)

        model_list['dp_model'][name] = dict()
        model_list['dp_model'][name]['model'] = model_file_path
        model_list['dp_model'][name]['peptide_length'] = peptide_length
        # model_list['max_x_length'] = res_map['max_x_length']

        del res_map
        gc.collect()
        K.clear_session()





    # save model data
    #file_all_models = open(out_dir + "/all_models.obj", 'wb')
    #pickle.dump(models, file_all_models)
    #file_all_models.close()

    ####################################################################################################################
    print("Ensemble learning ...")

    ## save result
    model_json = out_dir + "/model.json"
    with open(model_json, 'w') as f:
        json.dump(model_list, f)

    ## evaluation
    if test_file is not None:
        ptm_predict(model_json,test_file=test_file,db=db,out_dir=out_dir)
        #ensemble_predict(model_json,x=X_test,y=Y_test, batch_size=batch_size,method=ensemble_method,
        #                 out_dir=out_dir,
        #                 prefix="final_eval")

    ####################################################################################################################

def change_model(model, new_input_shape):
    # replace input shape of first layer
    """
    Used by AutoRT
    :param model:
    :param new_input_shape:
    :return:
    """
    print("Base model ...")
    print(model.get_weights())
    model._layers[1].batch_input_shape = (None,new_input_shape[0],new_input_shape[1])

    # rebuild model architecture by exporting and importing via json
    new_model = tf.keras.models.model_from_json(model.to_json())


    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            print("layer: %s" % (layer.name))
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    new_model.compile(loss='mean_squared_error',
                  ## In this case, we cannot change the learning rate.
                  optimizer=model.optimizer)
                  #metrics=['mean_squared_error'])

    new_model.summary()

    print("New model ...")
    print(new_model.get_weights())
    return new_model

#def change_model_input_shape(model, new_input_shape):
#    model._layers[1].batch_input_shape = (None, new_input_shape[0], new_input_shape[1])
#    return model

def get_model_input_shape(model):
    return model._layers[1].batch_input_shape

def get_peptide_length_from_model(model):
    return model._layers[1].batch_input_shape[1]


def ptm_prediction_for_multiple_ptms(model_dir:str, test_file:str, db=None, out_dir="./", prefix="test",
                                     method = "average",add_model_explain=False, bg_data=None):
    model_folders = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
    res_files = dict()
    for m_dir in model_folders:
        model_file = os.path.join(m_dir, "model.json")
        ptm_name = os.path.basename(m_dir)
        o_dir = os.path.join(out_dir, ptm_name)
        if os.path.isdir(o_dir) is False:
            os.makedirs(o_dir)
        print("Predict modification site for PTM: %s, output folder: %s \n" % (ptm_name, o_dir))
        out_file = ptm_predict(model_file=model_file, test_file=test_file, db=db, out_dir=o_dir, prefix=prefix, method=method,
                               add_model_explain=add_model_explain, bg_data=bg_data)
        if os.path.isfile(out_file):
            res_files[ptm_name] = out_file
        else:
            print("No result for %s" % (ptm_name))

    res = pd.concat([add_ptm_column(f, ptm_name) for ptm_name, f in res_files.items()], axis=0)
    o_file = out_dir + "/" + str(prefix) + ".tsv"
    res.to_csv(o_file, sep="\t", index=False)


def ptm_predict(model_file:str, test_file:str, db=None, out_dir="./", prefix="test", method = "average",add_confidence_metric="fpr",add_model_explain=False, bg_data=None):

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

        res = dl_models_predict(model_file, test_file=test_file, db=db, out_dir=out_dir, prefix=prefix, add_model_explain=add_model_explain, bg_data=bg_data)
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


def dl_models_predict(model_file:str, test_file:str, db:str, batch_size=2048, out_dir="./", prefix="test", add_model_explain=False, bg_data=None):

    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)

    y_dp = np.zeros(0)

    bg_db = db
    if add_model_explain:
        if bg_data is None:
            default_bg_files = get_default_bg_file(os.path.dirname(model_file))
            if default_bg_files is not None:
                bg_data = generate_bg_samples(default_bg_files[0], n_each_class=100000, out_dir=out_dir)
                bg_db = default_bg_files[1]
        else:
            bg_data = generate_bg_samples(bg_data, n_each_class=100000, out_dir=out_dir)

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

        if add_model_explain:
            print("Run model interpretability analysis ...")
            if isinstance(bg_data,str):
                ## only use the first one
                bg_data_df, bg_site_data = processing_prediction_data(bg_data, db=bg_db, flank_length=flank_length)
                model_explainer(model, x=x, bg_data=bg_data_df, out_dir=out_dir, prefix=str(name))
            else:
                model_explainer(model, x=x, bg_data=bg_data, out_dir=out_dir, prefix=str(name))



        gc.collect()
        K.clear_session()

    return y_dp

def change_model_input_shape(model, new_input_shape):

    print("Input shape: %d, %d\n" % (new_input_shape[0], new_input_shape[1]))
    model._layers[0]._batch_input_shape = (None, new_input_shape[0], new_input_shape[1])

    # rebuild model architecture by exporting and importing via json
    new_model = tf.keras.models.model_from_json(model.to_json())

    new_model.summary()

    return new_model