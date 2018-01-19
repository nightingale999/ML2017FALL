import numpy as np
import pickle
import argparse
from keras.models import load_model

# parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('-mp', '--model_path', default='./models/model_try.h5')
parser.add_argument('-rp', '--result_path', default='marvyn_out.csv')
parser.add_argument('-td', '--test_data', default='../../data/test.data')
parser.add_argument('-tc', '--test_csv', default='../../data/test.csv')
args = parser.parse_args()

model_path = args.model_path
result_path = args.result_path
test_data = args.test_data
test_csv = args.test_csv

# Readfile
print("Reading File")
with open(test_data, 'rb') as f:
    pretestSound = pickle.load(f, encoding='latin1')
with open(test_csv) as f:
    target = f.readlines()

model = load_model(model_path)
target_token_index = pickle.load(open('./temp_target_token_index.p', 'rb'))

# Preprocess
print("Preprocessing")
input_texts = []
target_texts = []
for line in target:
    line = line[:-1]
    sentences = line.split(',')
    for i in range(4):
        sentence = sentences[i].split() + ['\n']
        target_texts.append(sentence)
testSound = []
for sound in pretestSound:
	for i in range(4):
		testSound.append(sound)

max_encoder_seq_length = 246
max_decoder_seq_length = 15

encoder_input_data = np.zeros((len(testSound), max_encoder_seq_length, 39), dtype='float32')
decoder_input_data = np.zeros((len(testSound), max_decoder_seq_length), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(testSound, target_texts)):
    encoder_input_data[i, -len(input_text):, :] = input_text
    decoder_input_data[i, 0] =  target_token_index['\t']
    for t, char in enumerate(target_text):
        try:
            decoder_input_data[i, t+1] = target_token_index[char]
        except:
            decoder_input_data[i, t+1] = 0

result = model.predict([encoder_input_data, decoder_input_data])
print("Predicting...")
f = open(result_path, 'w')
f.write('id,answer\n')
for i in range(2000):
    s = i*4
    score = np.array(result[s:s+4]).reshape((4, 15)).mean(axis=1).reshape(-1)
    ans = np.argmax(score)
    f.write(str(i+1)+','+str(ans)+'\n')
