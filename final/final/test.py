import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='')
parser.add_argument('-gpu', '--use_gpu', action='store_true')
parser.add_argument('-cmp', '--c_model_path', default='./c_model_00')
parser.add_argument('-smp', '--s_model_path', default='./s_model_00')
parser.add_argument('-pmp', '--p_model_path', default='./p_model_00')
parser.add_argument('-sp', '--test_sound_path', default='./data/test.data')
parser.add_argument('-cp', '--test_caption_path', default='./data/test.csv')
parser.add_argument('-dp', '--dict_path', default='./char2id.pickle')
parser.add_argument('-rp', '--result_path', default='out.csv')

parser.add_argument('-msl', '--max_sound_length', type=int, default=246)
parser.add_argument('-mcl', '--max_caption_length', type=int, default=14)

parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-rl', '--rnn_layer', type=int, default=2)
parser.add_argument('-bi', '--bidirectional', action='store_true')

parser.add_argument('-ed', '--embedding_dim', type=int, default=100)
parser.add_argument('-hd', '--hidden_dim', type=int, default=100)
parser.add_argument('-dr', '--dropout_rate', type=float, default=0.5)

class caption_model(nn.Module):
    def __init__(self, args):
        super(caption_model, self).__init__()
        self.hidden_dim = test_sound_pathhidden_dim
        self.batch_size = args.batch_size
        self.max_caption_length = args.max_caption_length
        self.use_gpu = args.use_gpu

        self.word_embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.drop = nn.Dropout(args.dropout_rate)
        self.lstm = nn.LSTM(args.embedding_dim, args.hidden_dim, args.rnn_layer, dropout=args.dropout_rate, bidirectional=args.bidirectional, batch_first=True)
        self.c = args.rnn_layer
        if args.bidirectional:
            self.c *= 2
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        x = self.word_embedding(sentence)
        x = self.drop(x)
        out, self.hidden = self.lstm(x, self.hidden)
        return out[:, -1, :]

class sound_model(nn.Module):
    def __init__(self, args):
        super(sound_model, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.max_sound_length = args.max_sound_length
        self.use_gpu = args.use_gpu

        self.lstm = nn.LSTM(39, args.hidden_dim, args.rnn_layer, dropout=args.dropout_rate, bidirectional=args.bidirectional, batch_first=True)
        self.c = args.rnn_layer
        if args.bidirectional:
            self.c *= 2
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.c, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, mfcc):
        out, self.hidden = self.lstm(mfcc, self.hidden)
        return out[:, -1, :]

class pred_model(nn.Module):
    def __init__(self, args):
        super(pred_model, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sound, caption):
        dot_product = torch.sum(sound*caption, 1)
        out = self.sigmoid(dot_product)
        return out

def read_dict(args):
    print('read_dict')
    with open(args.dict_path, 'rb') as f:
        char2id = pickle.load(f)   
    args.vocab_size = len(char2id)
    return args, char2id

def read_test_file(args):
    print('read_test_file')
    with open(args.test_sound_path, 'rb') as f:
        test_sound = pickle.load(f, encoding='latin1')
    with open(args.test_caption_path, 'r') as f:
        test_caption = f.readlines()
    return test_sound, test_caption

def preprocess_test_data(args, sound, caption):
    print('preprocess_test_data')
    num_data = len(sound)
    sound_data = np.zeros((num_data, args.max_sound_length, 39), dtype='float32')
    for i, s in enumerate(sound):
        sound_data[i, -len(s):, :] = s

    return sound_data, num_data

def test(args):
    args, char2id = read_dict(args)
    c_model = torch.load(args.c_model_path)
    s_model = torch.load(args.s_model_path)
    p_model = torch.load(args.p_model_path)

    if args.use_gpu:
        c_model = c_model.cuda()
        s_model = s_model.cuda()
        p_model = p_model.cuda()

    c_model.eval()
    s_model.eval()
    p_model.eval()
    
    test_sound, test_caption = read_test_file(args)
    test_sound_data, test_ndata = preprocess_test_data(args, test_sound, test_caption)

    with open(args.result_path, 'w') as f:
        f.write('id,answer\n')

        for i in range(test_ndata):
            if i % 100 == 0:
                print(str(i)+'/'+str(test_ndata))

            sound_vectors = np.zeros((args.batch_size, args.max_sound_length, 39), dtype='float32')
            sound_vectors[0] = test_sound_data[i] 
            sound_vectors[1] = test_sound_data[i] 
            sound_vectors[2] = test_sound_data[i] 
            sound_vectors[3] = test_sound_data[i] 

            caption_vectors = np.zeros((args.batch_size, args.max_caption_length))
            captions = test_caption[i][:-1].split(',')

            for j, cap in enumerate(captions):
                cap_chars = cap.split(' ')
                cap_id = []
                for cc in cap_chars:
                    if cc in char2id:
                        cap_id.append(char2id[cc])
                cap_id = np.array(cap_id)
                if len(cap_id) > 0:
                    caption_vectors[j, -len(cap_id):] = cap_id

            s_input = Variable(torch.FloatTensor(sound_vectors))
            c_input = Variable(torch.LongTensor(caption_vectors))
            if args.use_gpu:
                s_input = s_input.cuda()
                c_input = c_input.cuda()

            s_model.hidden = s_model.init_hidden()
            c_model.hidden = c_model.init_hidden()
            s_output = s_model(s_input)
            c_output = c_model(c_input)
            p_output = p_model(s_output, c_output)

            pred = p_output.cpu().data.numpy()
            si = np.argmax(pred[:4])
            f.write(str(i+1)+','+str(si)+'\n')

def main():
    args = parser.parse_args()
    test(args)

if __name__ == '__main__':
    main()
