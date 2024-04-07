from lib.DataIO import variant_annotation_vcf, filter_db, getTrainDataFromFasta, getTrainDataFromPhosphoSitePlusFasta, \
    get_type_of_input_ptm_site_db, getTrainDataFromTable
from lib.MutationUtils import mutation_impact_prediction, mutation_impact_prediction_for_multiple_ptms
from lib.PTModels import ensemble_models, ptm_prediction_for_multiple_ptms, ptm_predict
import argparse
import sys
import os

def main():

    if len(sys.argv) == 1:
        print("python deepvep.py [train, predict, process, tool, vcf]")
        sys.exit(0)
    else:

        mode = sys.argv[1]

        if mode == "train":

            print("Run training!")
            parser = argparse.ArgumentParser(
                description='DeepVEP')
            parser = argparse.ArgumentParser(
                description='DeepVEP')
            parser.add_argument('-i', '--input', default=None, type=str, required=True,
                                help="Input data for training")
            parser.add_argument('-t', '--test', default=None, type=str,
                                help="Input data for testing")

            parser.add_argument('-d', '--db', default=None, type=str,
                                help="Protein database")

            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-e', '--epochs', default=20, type=int)
            parser.add_argument('-b', '--batch_size', default=128, type=int)
            parser.add_argument('-gpu', '--gpu_n', default=1, type=int)

            ## used for transfer learning
            parser.add_argument('-m', '--model_file', default=None, type=str)

            parser.add_argument('-g', '--ga', default=None, type=str)

            parser.add_argument('-n', '--early_stop_patience', default=None, type=int)
            parser.add_argument('-np', '--n_patience', default=5, type=int)
            parser.add_argument('-lr', '--learning_rate', default=None, type=float)

            # add_ReduceLROnPlateau
            parser.add_argument('-rlr', '--add_ReduceLROnPlateau', action='store_true')

            parser.add_argument('-c', '--add_eval_callback', action='store_true')
            parser.add_argument('-tb', '--add_tensorboard', action='store_true')

            parser.add_argument('-se', '--sequence_encode_method', default="one_hot", type=str,
                                help="Peptide encoding method, only one_hot is supported.")

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            input_file = args.input
            test_file = args.test
            out_dir = args.out_dir
            db = args.db

            epochs = args.epochs
            batch_size = args.batch_size
            n_gpu = args.gpu_n

            model_file = args.model_file
            ga = args.ga

            early_stop_patience = args.early_stop_patience
            add_ReduceLROnPlateau = args.add_ReduceLROnPlateau
            n_patience = args.n_patience
            learning_rate = args.learning_rate
            add_eval_callback = args.add_eval_callback

            add_tb_callback = args.add_tensorboard

            sequence_encode_method = args.sequence_encode_method

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            ensemble_models(input_data=input_file, test_file=test_file, db=db, nb_epoch=epochs, batch_size=batch_size,
                            ga_file=ga, out_dir=out_dir, early_stop_patience=early_stop_patience,models_file=model_file,
                            add_ReduceLROnPlateau=add_ReduceLROnPlateau, add_eval_callback=add_eval_callback,
                            add_tb_callback=add_tb_callback,gpu=n_gpu,n_patience=n_patience,lr=learning_rate,seq_encode_method=sequence_encode_method)

        elif mode == "predict":

            print("Run prediction!")
            parser = argparse.ArgumentParser(
                description='DeepVEP')
            parser.add_argument('-i', '--input', default=None, type=str, required=False,
                                help="Input data for prediction")
            parser.add_argument('-d', '--db', default=None, type=str, required=False,
                                help="Protein database")
            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-w', '--window_size', default=7, type=int,
                                help="Window size")

            parser.add_argument('-m', '--model', default=None, type=str, required=False, help="Trained general model")
            parser.add_argument('-t', '--task', default=1, type=int, help="Prediction type: 1=Mutation impact prediction, 2=Peptide detectability or phosphorylation.")

            parser.add_argument('-e', '--ensemble', default=1, type=int,
                                help="Ensemble method, 1: average, 2: meta_lr, default is 1.")

            ## model interpretability
            parser.add_argument('-s', '--explain_model', action='store_true')
            parser.add_argument('-b', '--bg_data',  default=None, type=str, required=False,
                                help="Data used as background data in model interpretability")


            if len(sys.argv) == 2:
                parser.print_usage()
                sys.exit(0)

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            prediction_type = args.task
            input_file = args.input
            out_dir = args.out_dir
            model_file = args.model

            window_size = args.window_size
            #flank_length = args.flank_length
            db = args.db
            #ptm_type = args.ptm

            ensemble_method_code = args.ensemble
            ensemble_method = "average"
            if ensemble_method_code == 1:
                ensemble_method = "average"
            elif ensemble_method_code == 2:
                ensemble_method = "meta_lr"


            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            ## for model interpretability
            add_model_explain = args.explain_model
            ## The format is the same as data file for "-i"
            bg_data = args.bg_data

            # Mutation impact prediction
            if prediction_type == 1:
                if model_file is None:
                    model_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "models")
                    if os.path.isdir(model_dir):
                        mutation_impact_prediction_for_multiple_ptms(model_dir, input_file=input_file, db=db,
                                                                     window_size=window_size, out_dir=out_dir,
                                                                     ensemble_method=ensemble_method,
                                                                     add_model_explain=add_model_explain,
                                                                     bg_data=bg_data)
                    else:
                        print("Model folder doesn't exist: %s" % (model_dir))
                        sys.exit(1)
                else:
                    if os.path.isdir(model_file):
                        mutation_impact_prediction_for_multiple_ptms(model_file, input_file=input_file, db=db,
                                                                     window_size=window_size, out_dir=out_dir,
                                                                     ensemble_method=ensemble_method,
                                                                     add_model_explain=add_model_explain,
                                                                     bg_data=bg_data)
                    else:
                        mutation_impact_prediction(model_file, input_file=input_file, db=db, window_size=window_size,
                                                   out_dir=out_dir, ensemble_method=ensemble_method,
                                                   add_model_explain=add_model_explain,
                                                   bg_data=bg_data)

            # PTM prediction
            elif prediction_type == 2:
                if model_file is None:
                    model_dir = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "models")
                    if os.path.isdir(model_dir):
                        ptm_prediction_for_multiple_ptms(model_dir=model_dir, test_file=input_file, db=db,
                                                         out_dir=out_dir, prefix="site_prediction",
                                                         add_model_explain=add_model_explain,
                                                         bg_data=bg_data)
                    else:
                        print("Model folder doesn't exist: %s" % (model_dir))
                        sys.exit(1)
                else:
                    if os.path.isdir(model_file):
                        ptm_prediction_for_multiple_ptms(model_dir=model_file, test_file=input_file, db=db,
                                                         out_dir=out_dir, prefix="site_prediction",
                                                         add_model_explain=add_model_explain,
                                                         bg_data=bg_data)
                    else:
                        ptm_predict(model_file=model_file, test_file=input_file, db=db, out_dir=out_dir,
                                    prefix="site_prediction",
                                    add_model_explain=add_model_explain,
                                    bg_data=bg_data)

        elif mode == "process":
            print("Run process!")
            # prepare data for training, testing and prediction
            parser = argparse.ArgumentParser(
                description='DeepVEP')
            parser.add_argument('-i', '--input', default=None, type=str, required=True,
                                help="Input data")
            parser.add_argument('-r', '--refine', default=None, type=str, required=False,
                                help="Potential sites. A file with similar format to --input and is used to refine negative samples.")
            parser.add_argument('-ptm', '--ptm', default="sty", type=str, required=True,
                                help="modification site, default is phosphorylation: sty")
            parser.add_argument('-d', '--db', default=None, type=str,
                                help="Protein database")
            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-f', '--flank_length', default=15, type=int,
                                help="The flank length, default is 15")
            parser.add_argument('-e', '--enzyme', default=None, type=str,
                                help="Protein digestion enzyme, such as trypsin")
            parser.add_argument('-m', '--missed_cleavages', default=2, type=int,
                                help="The max missed cleavages, default is 2")
            parser.add_argument('-l1', '--min_peptide_length', default=7, type=int,
                                help="Min digested peptide length")
            parser.add_argument('-l2', '--max_peptide_length', default=45, type=int,
                                help="Max digested peptide length")

            parser.add_argument('-n', '--min_n', default=10, type=int,
                                help="Consider negative sites from those proteins with >= n known sites.")

            parser.add_argument('-s', '--split_ratio', default=0.1, type=float,
                                help="Float between 0 and 1. Fraction of the input data to be used as testing data.")


            if len(sys.argv) == 2:
                parser.print_usage()
                sys.exit(0)

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            input_file = args.input
            refine_file = args.refine
            out_dir = args.out_dir
            flank_length = args.flank_length
            enzyme = args.enzyme
            max_missed_cleavages = args.missed_cleavages
            min_peptide_length = args.min_peptide_length
            max_peptide_length = args.max_peptide_length
            ptm_type = args.ptm

            split_ratio = args.split_ratio
            min_n = args.min_n

            # Optional, this is a protein database in FASTA format
            db = args.db

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if get_type_of_input_ptm_site_db(input_file, ptm_type=ptm_type) == 1:
                getTrainDataFromFasta(db=input_file, ptm_type=ptm_type, out_dir=out_dir, flank_length=flank_length,
                                                     enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                                                     min_peptide_length=min_peptide_length,
                                                     max_peptide_length=max_peptide_length,min_n=min_n)
            elif get_type_of_input_ptm_site_db(input_file, ptm_type=ptm_type) == 2:
                getTrainDataFromPhosphoSitePlusFasta(db=input_file, out_dir=out_dir, flank_length=flank_length,
                                                     enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                                                     min_peptide_length=min_peptide_length,
                                                     max_peptide_length=max_peptide_length,min_n=min_n)
            elif get_type_of_input_ptm_site_db(input_file, ptm_type=ptm_type) == 3:
                # The input for -i is a txt table which contains the protein ID and modification position
                getTrainDataFromTable(db=db, site_table=input_file, ptm_type=ptm_type,
                                      out_dir=out_dir, flank_length=flank_length,
                                      enzyme=enzyme, max_missed_cleavages=max_missed_cleavages,
                                      min_peptide_length=min_peptide_length,
                                      max_peptide_length=max_peptide_length,min_n=min_n,
                                      split_ratio=split_ratio,refine_file=refine_file)

        elif mode == "tool":

            parser = argparse.ArgumentParser(
                description='DeepVEP')
            parser.add_argument('-i', '--input', default=None, type=str,
                                help="Input data")
            parser.add_argument('-p', '--pattern', default=None, type=str,
                                help="The pattern to filter input data")
            parser.add_argument('-d', '--db', default="./", type=str,
                                help="Protein database")

            parser.add_argument('-o', '--output', default=None, type=str,
                                help="Output file or folder")

            if len(sys.argv) == 2:
                parser.print_usage()
                sys.exit(0)

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            pattern_str = args.pattern
            db = args.db
            input_file = args.input


            if (db is not None) and (pattern_str is not None):
                out_file = args.output
                filter_db(db, pattern_str, out_db=out_file)

        elif mode == "vcf":

            print("VCF annotation!")
            parser = argparse.ArgumentParser(
                description='DeepVEP')
            parser.add_argument('-i', '--input', default=None, type=str,
                                help="Input data")
            parser.add_argument('-c', '--config', default=None, type=str,
                                help="The pattern to filter input data")
            parser.add_argument('-o', '--output', default=None, type=str,
                                help="Output folder")

            if len(sys.argv) == 2:
                parser.print_usage()
                sys.exit(0)

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            vcf_file = args.input
            config_file = args.config
            out_dir = args.output

            variant_annotation_vcf(vcf_file,config_file,out_dir)


if __name__=="__main__":
    main()
