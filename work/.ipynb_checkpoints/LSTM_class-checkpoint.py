import re, sys # import des regex
import numpy as np # import des math
import tensorflow as tf
from elephas.utils.rdd_utils import to_simple_rdd

class Sequences():
    def __init__(self, text_object, max_len, step):
        self.tokens_ind = text_object.tokens_ind
        self.max_len = max_len
        self.step = step
        self.sequences, self.next_words = self.create_sequences()
    
    # affichage associé a la class Sequence
    def __repr__(self):
        # return 'Sequence object of max_len: {} and step: {}'.format(self.max_len, self.step)
        return 'Sequence object of max_len: %d and step: %d' % (self.max_len, self.step)

    # totale des sequences
    def __len__(self):
        return len(self.sequences)

    # decouteur des token en sequence avec une taille et un pas donnée
    def create_sequences(self):
        sequences = []
        next_words = []
        
        for i in range(0, len(self.tokens_ind) - self.max_len, self.step): # parcour toute la liste des tokens
            sequences.append(self.tokens_ind[i: i + self.max_len]) # ajoute la sequence
            next_words.append(self.tokens_ind[i + self.max_len]) # ajoute le prochain mot associé
        return sequences, next_words

    # affichage informatif
    def sequences_info(self):
        # print('number of sequences of length {}: {}'.format(self.max_len, len(self.sequences)))
        print('number of sequences of length %d: %d' % (self.max_len, len(self.sequences)))


class ModelPredict():
    # initiallisation 
    def __init__(self, model, prefix, token2ind, ind2token, max_len, embedding=False):
        self.model = model
        self.token2ind, self.ind2token = token2ind, ind2token
        self.max_len = max_len
        self.prefix = prefix
        self.tokens_ind = prefix.tokens_ind.copy()
        self.embedding = embedding

    # affichage associé a la class
    def __repr__(self):
        return self.prefix.content
    
    # cree un masque avec la sequence entre pour la prediction
    def single_data_generation(self):
        single_sequence = np.zeros((1, self.max_len, len(self.token2ind)), dtype=np.bool) # remplie avec true / false
        prefix = self.tokens_ind[-self.max_len:] # 

        for i, s in enumerate(prefix):
            single_sequence[0, i, s] = 1
        return single_sequence

    # on recupere la sequence 0 predite par notre model
    def model_predict(self):
        if self.embedding: # a savoir si embedding = True ou pas
            model_input = np.array(self.tokens_ind).reshape(1,-1) # redimension pour que notre vecteur soit accepté
        else: # la dimension etant correcte 
            model_input = self.single_data_generation() # on cree un masque sur notre sequence d'entrée
        return self.model.predict(model_input)[0] # on predit la sequence suivante

    @staticmethod
    def add_prob_temperature(prob, temperature=1):
        prob = prob.astype(float) # conversion en float pour etre sur
        # proba associé a une temperature
        prob_with_temperature = np.exp(np.where(prob == 0, 0, np.log(prob + 1e-10)) / temperature)
        prob_with_temperature /= np.sum(prob_with_temperature) # on divise la somme de toute les proba recuperer
        return prob_with_temperature

    @staticmethod
    def reverse_preprocess(text):
        text_reverse = re.sub(r'\s+(["\'().,;-])', r'\1', text)
        text_reverse = re.sub(' +', ' ', text_reverse)
        return text_reverse

    
    def return_next_word(self, temperature=1, as_word=False):
        prob = self.model_predict()
        
        prob_with_temperature = self.add_prob_temperature(prob, temperature)
        # tire aleatoirement une proba associé au prochain mot dans la prediction
        next_word = np.random.choice(len(prob_with_temperature), p=prob_with_temperature)
        
        if as_word:
            return self.ind2token[next_word] # on recupere le mot associé a une proba
        else:
            return next_word # on recupere la proba

    # generation de la sequence n+1
    def generate_sequence(self, k, append=False, temperature=1):
        """
        k: nombre de sequence a predire
        """
        for i in range(k):
            # on recupere la proba du prochain mot le plus probable selon une temperature
            next_word = self.return_next_word(temperature=temperature)
            self.tokens_ind.append(next_word) # on append la proba
        return_tokens_ind = self.tokens_ind # copy sur une variable interne
        # pour chaque proba on recupere le mot associé
        return_tokens_ind = ' '.join([self.ind2token[ind] for ind in return_tokens_ind])

        if not append:
            self.tokens_ind = self.prefix.tokens_ind.copy()
        # on reverse le preprocess pour avoir la ponctuation
        return self.reverse_preprocess(return_tokens_ind)

    # effectue une serie de prediction
    def bulk_generate_sequence(self, k, n, temperature=1):
        for i in range(n):
            print(self.generate_sequence(k, temperature=temperature))
            print('\n')


class TextDataGenerator(tf.keras.utils.Sequence):
    # initialisation
    def __init__(self, spark, sequences, next_words, sequence_length, vocab_size, batch_size=32, shuffle=True, embedding=False):
        self.batch_size = batch_size
        self.sequences = sequences
        self.next_words = next_words
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.shuffle = shuffle
        self.spark_ = spark
        self.context = None
        self.embedding = embedding
        self.on_epoch_end()
    
    # recupere la proportion (seq/batch) en entier
    def __len__(self):
        return int(np.floor(len(self.sequences) / self.batch_size))

    # creeation des batch a chaque epoch
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        sequences_batch = [self.sequences[k] for k in indexes]
        next_words_batch = [self.next_words[k] for k in indexes]

        if self.embedding:
            X = np.array(sequences_batch)
            y = tf.keras.utils.to_categorical(next_words_batch, num_classes=self.vocab_size)
        else:
            X, y = self.__data_generation(sequences_batch, next_words_batch)
        return X, y
    
    # generer une grosse rdd contenant toute les sequences
    def generate_rdds(self):
        X = np.array(self.sequences)
        y = tf.keras.utils.to_categorical(self.next_words, num_classes=self.vocab_size)
        return to_simple_rdd(self.spark_, X, y)
    
    # get une single batch
    def generate_1_rdd(self, index=1):
        indexes = self.indexes[0: (index + 1) * self.batch_size]
        sequences_batch = [self.sequences[k] for k in indexes]
        next_words_batch = [self.next_words[k] for k in indexes]

        if self.embedding:
            X = np.array(sequences_batch)
            y = tf.keras.utils.to_categorical(next_words_batch, num_classes=self.vocab_size)
        else:
            X, y = self.__data_generation(sequences_batch, next_words_batch)
        return to_simple_rdd(self.spark_, X, y)

    # Le brassage prend les extrémités de ces séries temporelles à partir d'emplacements aléatoires
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # cree un masque bool sur notre texte d'entrée
    def __data_generation(self, sequences_batch, next_words_batch):
        X = np.zeros((self.batch_size, self.sequence_length, self.vocab_size), dtype=np.bool)
        y = np.zeros((self.batch_size, self.vocab_size), dtype=np.bool)

        for i, seq in enumerate(sequences_batch):
            for j, word in enumerate(seq):
                X[i, j, word] = 1
                y[i, next_words_batch[i]] = 1
        return X, y