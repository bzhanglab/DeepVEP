import keras
from keras import ops
from keras import layers
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import sys
import os

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def train(train_file:str):
    ## Load the data, x is the sequences, y is the labels 
    # protein   aa  pos x   y   peptide
    # Q9HAH7    T   330 AAAAAAAAAAAAAAATGPQGLHLLFERPRPP 0   -
    # P58012    S   238 AAAAAAAAAAAAGPGSPGAAAVVKGLAGPAA 0   -
    # O75400    S   299 AAAAAAAAAAANANASTSASNTVSGTVPVVP 0   -
    # Load and preprocess the data
    df = pd.read_csv(train_file, sep='\t')
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # Generate training and validation data
    ## randomly select the same number of positive (df['y']==1) and negative (df['y']==0) samples
    n_1 = df[df['y']==1].shape[0]
    n_0 = df[df['y']==0].shape[0]

    df_1 = df[df['y']==1]
    df_0 = df[df['y']==0].sample(n=n_1, random_state=42)
    df = pd.concat([df_1, df_0], axis=0)
    x_train, x_val, y_train, y_val = train_test_split(df['x'], df['y'], test_size=0.2, random_state=42)

    # One-hot encode the labels
    #y_train = to_categorical(y_train, num_classes=2)
    #y_val = to_categorical(y_val, num_classes=2)

    # Get the max length of the sequences in column x
    maxlen = max(df['x'].apply(lambda x: len(x)))

    # Get all unique characters in the sequences x
    vocab_size = len(set(''.join(df['x'])))

    # Tokenize the sequences for model.fit
    tokenizer = Tokenizer(num_words=vocab_size, char_level=True)
    tokenizer.fit_on_texts(df['x'])

    x_train = tokenizer.texts_to_sequences(x_train)
    x_val = tokenizer.texts_to_sequences(x_val)

    # Pad the sequences
    x_train = pad_sequences(x_train, maxlen=maxlen)
    x_val = pad_sequences(x_val, maxlen=maxlen)

    embed_dim = 64  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Use sigmoid for binary classification

    model = models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train and evaluate the model
    history = model.fit(
        x_train, y_train,
        batch_size=512,
        epochs=40,
        validation_data=(x_val, y_val)
    )
    return [tokenizer,maxlen,model]

if __name__ == '__main__':

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    tokenizer,maxlen,model = train(train_file)
    model.save('my_model.keras')

    df = pd.read_csv(test_file, sep='\t')
    # convert df['x'] to numpy array
    x_test = tokenizer.texts_to_sequences(df['x'])
    x_test = pad_sequences(x_test, maxlen=maxlen)
    y_pred = model.predict(x_test)
    df['y_pred'] = y_pred
    df.to_csv('test_result.tsv', index=False, sep='\t')
