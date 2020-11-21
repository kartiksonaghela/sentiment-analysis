import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf

#loading the train dataset
df = pd.read_csv("Train.csv")
data_test=pd.read_csv("test.csv")

(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=df,
                                                                   text_column = 'tweet',
                                                                   label_columns = 'label',
                                                                   val_df = data_test,
                                                                   maxlen = 500,
                                                                   preprocess_mode = 'bert')
                                                                   
model = text.text_classifier(name = 'bert',
                             train_data = (X_train, y_train),
                             preproc = preproc)
                             

learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 6)

learner.fit_onecycle(lr = 2e-5, epochs = 1)

predictor = ktrain.get_predictor(learner.model, preproc)

predictor.save('final')
