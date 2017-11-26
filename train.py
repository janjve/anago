import os
import anago
from anago.data.reader import load_data_and_labels, load_word_embeddings
from anago.data.preprocess import prepare_preprocessor
from anago.config import ModelConfig, TrainingConfig
from anago.models import SeqLabeling

DATA_ROOT = 'data/conll2003/en/ner'
SAVE_ROOT = './models'  # trained model
LOG_ROOT = './logs'     # checkpoint, tensorboard
embedding_path = '/media/jan/OS/Dataset/WordEmbeddings/wiki.en.vec'
model_config = ModelConfig()
training_config = TrainingConfig()

model_path = os.path.join(SAVE_ROOT, 'mymodel.h5')

train_path = os.path.join(DATA_ROOT, 'train.small.txt')
valid_path = os.path.join(DATA_ROOT, 'valid.small.txt')

x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)

p = prepare_preprocessor(x_train, y_train)
embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)
model_config.vocab_size = len(p.vocab_word)
model_config.char_vocab_size = len(p.vocab_char)

model = SeqLabeling(model_config, embeddings, len(p.vocab_tag))
trainer = anago.Trainer(model,
                        training_config,
                        checkpoint_path=LOG_ROOT,
                        save_path=SAVE_ROOT,
                        preprocessor=p,
                        embeddings=embeddings)
trainer.train(x_train, y_train, x_valid, y_valid)
evaluator = anago.Evaluator(model, preprocessor=p)
model.save(model_path)
