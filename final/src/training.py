from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout, TimeDistributed
from keras.layers import Activation, BatchNormalization, LeakyReLU
from keras.optimizers import *
from keras.callbacks import *
import numpy as np
import pickle
import argparse
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# set the memory used in GPU to 50%
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('-ep', '--epochs', type=int, default=100)
parser.add_argument('-bs', '--batch_size', type=int, default=512)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-ed', '--embedding_dim', type=int, default=256)
parser.add_argument('-hd', '--hidden_dim', type=int, default=256)
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5)
parser.add_argument('-mp', '--model_path', default='./models/model_try.h5')
parser.add_argument('-ds', '--data_size', type=int, default = 5)
parser.add_argument('-la', '--leaky_alpha', type=float, default = 0.5)
parser.add_argument('-dw', '--dense_width', type=int, default = 16)
parser.add_argument('-td', '--train_data', default='../../data/train.data')
parser.add_argument('-tc', '--train_caption', default='../../data/train.caption')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
dropout_rate = args.dropout_rate
model_path = args.model_path
opt = RMSprop(lr = args.learning_rate)
data_size = args.data_size
leaky_alpha = args.leaky_alpha
dense_width = args.dense_width
train_data = args.train_data
train_caption = args.train_caption

num_decoder_tokens = 0
max_decoder_seq_length = 0
max_encoder_seq_length = 0

def preprocess():
    global num_decoder_tokens
    global max_decoder_seq_length
    global max_encoder_seq_length
    # Read File
    print("Reading File")
    with open(train_data, 'rb') as f:
        trainSound = pickle.load(f)
    with open(train_caption) as f:
        target = f.readlines()

    # Preprocess
    print("Preprocessing")
    input_texts = []
    target_texts = []
    target_characters = []
    for line in target:
        sentence = line.split(' ')
        for character in sentence:
            if character not in target_characters:
                target_characters.append(character)
        target_texts.append(sentence)

    target_characters.append('\t')
    max_decoder_seq_length = max([len(t.split(' ')) for t in target]) + 1
    input_characters = trainSound
    target_characters = sorted(target_characters)

    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(sound) for sound in trainSound])

    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    # Build training data
    encoder_input_data_origin = np.zeros((len(trainSound), max_encoder_seq_length, 39), dtype='float32')
    decoder_input_data_origin = np.zeros((len(trainSound), max_decoder_seq_length), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(trainSound, target_texts)):
        encoder_input_data_origin[i, -len(input_text):, :] = input_text
        decoder_input_data_origin[i, 0] =  target_token_index['\t']
        for t, char in enumerate(target_text):
            decoder_input_data_origin[i, t+1] = target_token_index[char]

    ans = np.ones((len(trainSound), 15, 1))
    encoder_input_data = encoder_input_data_origin
    decoder_input_data = decoder_input_data_origin

    for k in range(data_size):
        temp_encoder_input_data = encoder_input_data_origin
        temp_decoder_input_data = np.roll(decoder_input_data_origin, k+1, axis=0)
        temp_ans = np.zeros((len(trainSound), 15, 1))
        encoder_input_data = np.concatenate((encoder_input_data, temp_encoder_input_data), axis=0)
        decoder_input_data = np.concatenate((decoder_input_data, temp_decoder_input_data), axis=0)
        ans = np.concatenate((ans, temp_ans), axis=0)

    print("Training_data_size:", encoder_input_data.shape[0])
    
    L = np.random.permutation(encoder_input_data.shape[0])
    n = int(0.9 * encoder_input_data.shape[0])
    encoder_input_data_train = encoder_input_data[L[:n]]
    decoder_input_data_train = decoder_input_data[L[:n]]
    ans_train = ans[L[:n]]
    encoder_input_data_validation = encoder_input_data[L[n:]]
    decoder_input_data_validation = decoder_input_data[L[n:]]
    ans_validation = ans[L[n:]]
    return encoder_input_data_train, decoder_input_data_train, ans_train, encoder_input_data_validation, decoder_input_data_validation, ans_validation

def build_model():
    # Build Model
    print("Building Model")
    encoder_inputs = Input(shape=(None, 39))
    encoder_lstm = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=num_decoder_tokens+1, output_dim=embedding_dim)
    #decoder_embedding = Dense(embedding_dim)
    decoder_vectors = decoder_embedding(decoder_inputs)
    decoder_dropouts = Dropout(dropout_rate)(decoder_vectors)
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    #decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs, _, _ = decoder_lstm(decoder_dropouts, initial_state=encoder_states)
    evaluate_dense = TimeDistributed(Dense(dense_width))(decoder_outputs)
    evaluate_dense = TimeDistributed(LeakyReLU(alpha=leaky_alpha))(evaluate_dense)
    evaluate_dense = TimeDistributed(Dense(1, kernel_initializer='normal'))(evaluate_dense)
    evaluate_dense = TimeDistributed(BatchNormalization())(evaluate_dense)
    evaluate_dense_outputs = TimeDistributed(Activation('sigmoid'))(evaluate_dense)
    
    #decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    #decoder_dense_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], evaluate_dense_outputs)
    return model

def Train(model, EDT, DDT, AT, EDV, DDV, AV):
    # Training
    print("Training")
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience = 5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    history_callback = model.fit([EDT, DDT], AT,
     batch_size=batch_size,
     epochs=epochs,
     validation_data = [[EDV, DDV], AV],
     verbose=1,
     callbacks=[checkpoint, earlystop]
    )

def main():
    EDT, DDT, AT, EDV, DDV, AV = preprocess()
    model = build_model()
    Train(model, EDT, DDT, AT, EDV, DDV, AV)

if __name__=='__main__':
	main()
