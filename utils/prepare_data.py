############################################ IMPORT ##########################################################
import sys, os, json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

############################################ LOAD W2V EMBEDDING ###############################################
def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nLoading embeddings...')

    words = set()
    with open(train_file_path, 'r') as json_file:
        data = json.load(json_file)
        for conversation in data:
            for utterance in conversation['conversation']:
                text = utterance['text'].lower()
                words.update(text.split())

    words = list(words)  # Convert words set to a list
    word_idx = {word: idx + 1 for idx, word in enumerate(words)}  # Create word indices
    word_idx_rev = {idx + 1: word for idx, word in enumerate(words)}  # Reverse word indices

    w2v = {}
    with open(embedding_path, 'r') as embedding_file:
        embedding_file.readline()  # Skip the header line
        for line in embedding_file:
            line = line.strip().split(' ')
            w, ebd = line[0], line[1:]
            w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for word in words:
        if word in w2v:
            vec = list(map(float, w2v[word]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)

    print(f'Embedding file: {embedding_path}\nAll words: {len(words)} Hit words: {hit}')
    embedding.extend([list(np.random.rand(embedding_dim) / 5. - 0.1)] * (len(words) - hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])
    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print(f"embedding.shape: {embedding.shape} embedding_pos.shape: {embedding_pos.shape}")
    print("Loading embeddings done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos


############################################ LOAD DATA PAIR STEP ##############################################
def load_data_pair(input_file, word_idx, max_doc_len=75, max_sen_len=45):
    print('Loading data_file: {}'.format(input_file))
    pair_id_all, y_position, y_cause, y_pair, x, sen_len, doc_len, distance = [], [], [], [], [], [], [], []
    n_cut = 0

    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    for conversation in data:
        doc_id = conversation['conversation_ID']
        conversation_data = conversation['conversation']
        d_len = len(conversation_data)
        pairs = conversation['emotion-cause_pairs']
        pos_list = []
        cause_list = []

        for pair in pairs:
            emotion, cause = pair
            pos_list.append(int(emotion.split('_')[0]))
            cause_list.append(int(cause.split('_')[0]))

        pairs = [(pos_list[i], cause_list[i]) for i in range(len(pos_list))]
        pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])
        y_position_tmp, y_cause_tmp, y_pair_tmp, sen_len_tmp, x_tmp, distance_tmp = np.zeros((max_doc_len, 2)), np.zeros((max_doc_len, 2)), np.zeros((max_doc_len * max_doc_len, 2)), np.zeros((max_doc_len, )), np.zeros((max_doc_len, max_sen_len)), np.zeros((max_doc_len * max_doc_len, ))

        for i in range(d_len):
            utterance = conversation_data[i]
            words = utterance['text'].split()
            sen_len_tmp[i] = min(len(words), max_sen_len)
            for j, word in enumerate(words):
                word = word.lower()
                if j >= max_sen_len:
                    n_cut += 1
                    break
                if word not in word_idx:
                    x_tmp[i][j] = 24166
                else:
                    x_tmp[i][j] = word_idx[word]

        for i in range(d_len):
            for j in range(d_len):
                # Check whether i is an emotion clause
                if i + 1 in pos_list:
                    y_position_tmp[i][0] = 0
                    y_position_tmp[i][1] = 1
                else:
                    y_position_tmp[i][0] = 1
                    y_position_tmp[i][1] = 0
                # Check whether j is a cause clause
                if j + 1 in cause_list:
                    y_cause_tmp[j][0] = 0
                    y_cause_tmp[j][1] = 1
                else:
                    y_cause_tmp[j][0] = 1
                    y_cause_tmp[j][1] = 0
                # Check whether i, j clauses are emotion-cause pairs
                pair_id_curr = doc_id * 10000 + (i + 1) * 100 + (j + 1)
                if pair_id_curr in pair_id_all:
                    y_pair_tmp[i * max_doc_len + j][0] = 0
                    y_pair_tmp[i * max_doc_len + j][1] = 1
                else:
                    y_pair_tmp[i * max_doc_len + j][0] = 1
                    y_pair_tmp[i * max_doc_len + j][1] = 0
                # Find the distance between the clauses, and use the same embedding beyond 10 clauses
                distance_tmp[i * max_doc_len + j] = min(max(j - i + 100, 90), 110)

        y_position.append(y_position_tmp)
        y_cause.append(y_cause_tmp)
        y_pair.append(y_pair_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)
        doc_len.append(d_len)
        distance.append(distance_tmp)

    y_position, y_cause, y_pair, x, sen_len, doc_len, distance = map(torch.tensor, \
    [y_position, y_cause, y_pair, x, sen_len, doc_len, distance])

    for var in ['y_position', 'y_cause', 'y_pair', 'x', 'sen_len', 'doc_len', 'distance']:
        print('{}.shape {}'.format(var, eval(var).shape))
    print('n_cut {}'.format(n_cut))
    print('Loading data done!\n')
    return y_position, y_cause, y_pair, x, sen_len, doc_len, distance
