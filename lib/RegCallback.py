
from tensorflow.keras.callbacks import Callback
from .Metrics import evaluate_model



class RegCallback(Callback):

    def __init__(self, X_train, X_test, y_train, y_test, scale_para=None):
        self.x = X_train
        self.y = y_train
        self.x_val = X_test
        self.y_val = y_test

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):

        ## training data
        print("Training data ==>\n")
        y_pred = self.model.predict(self.x)
        evaluate_model(self.y, y_pred, plot=False)
        ## test data
        print("Testing data ==>\n")
        y_pred_val = self.model.predict(self.x_val)
        evaluate_model(self.y_val, y_pred_val, plot=False)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
