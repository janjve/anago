import os
import anago
from anago.data.reader import load_data_and_labels, load_word_embeddings
from anago.data.preprocess import prepare_preprocessor
from anago.config import ModelConfig, TrainingConfig
from anago.models import SeqLabeling
import numpy as np

DATA_ROOT = 'data/conll2003/en/ner'
LOAD_ROOT = './models'  # trained model
LOG_ROOT = './logs'     # checkpoint, tensorboard
embedding_path = '/media/jan/OS/Dataset/WordEmbeddings/wiki.en.vec'
model_config = ModelConfig()

test_path = os.path.join(DATA_ROOT, 'train.small.txt')
x_test, y_test = load_data_and_labels(test_path)

p = prepare_preprocessor(x_test, y_test)

embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)
model_config.vocab_size = len(p.vocab_word)
model_config.char_vocab_size = len(p.vocab_char)

model_path = os.path.join(LOAD_ROOT, 'mymodel.h5')
model = SeqLabeling(model_config, embeddings, len(p.vocab_tag))
model.load(model_path)
X, y = p.transform(x_test, y_test)
predictions = model.predict(X)

for words, prediction, sentence_length in zip(x_test, predictions, X[2]):
    nopad_prediction = prediction[:sentence_length.item()]
    label_indeces = [np.argmax(x) for x in nopad_prediction]
    labels = p.inverse_transform(label_indeces)

    print "\n".join(["{}\t{}".format(w, l) for w, l in zip(words, labels)])
    print ''
