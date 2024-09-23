'''
@author: Angxiao

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
'''
import xml.etree.cElementTree as ET
import codecs
import nltk
import torch

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download("brown")
nltk.download("treebank")
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import brown

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

import torch

import numpy as np

from scipy.sparse import vstack
import pickle
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

# Initialize Lemmatizer and Stop Words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer()
encoder = LabelEncoder()


class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id  # id of the WSD instance
        self.lemma = lemma  # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index  # index of lemma within the context
        self.baseline = "None"
        self.lesksynset = "None"

    def __str__(self):
        '''
        For printing purposes.
        '''
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)

def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            context = process_text(context)
            # print(context)
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    # print(my_id)
                    lemma = to_ascii(el.attrib['lemma'])
                    # print(lemma)
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances


def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        # print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        sense_key = [wn.lemma_from_key(lemma_key).synset().name() for lemma_key in
                     sense_key.split()]  # convert the lemma key to synset key
        if doc == 'd001':
            dev_key[my_id] = sense_key
        else:
            test_key[my_id] = sense_key
    return dev_key, test_key


def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')

def process_text(context):
    # Tokenize the sentence
    # words = word_tokenize(context)
    words = [word.lower() for word in context if word.isalpha()]
    # Remove stop words and lemmatize the remaining words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return words

def baseline(wsd_instance):
    count = {}
    synsets = wn.synsets(wsd_instance.lemma)

    for ss in synsets:
        freq = 0
        for lemma in ss.lemmas():
            freq += lemma.count()
        count[ss.name()] = freq
    max = 0
    ss = synsets[0]
    for ss_name, count in count.items():
        if count > max:
            max = count
            ss = wn.synset(ss_name)

    return ss


def leskpredict(wsd_instance):
    # print(wsd_instance.context)
    # print(wsd_instance.lemma)
    return lesk(wsd_instance.context, wsd_instance.lemma)

def predict(wsd_instance):
    baseline_synset = baseline(wsd_instance)
    lesk_synset = leskpredict(wsd_instance)
    return baseline_synset.name(), lesk_synset.name()



def find_the_instances(instances, w):
    instance_dict = {k:v for (k, v) in instances.items() if v.lemma in w }
    return instance_dict
def is_correct_prediction(predicted, correct_senses):
    return predicted is None or predicted in correct_senses


def evaluate(instances, keys):
    baseline_correct_predictions = 0
    lesk_correct_predictions = 0

    for instance_id, wsd_instance in instances.items():
        baseline_pred, lesk_pred = predict(wsd_instance)
        correct_senses = keys[instance_id]
        # print(correct_senses)
        if is_correct_prediction(baseline_pred, correct_senses):
            baseline_correct_predictions += 1
        if is_correct_prediction(lesk_pred, correct_senses):
            lesk_correct_predictions += 1
    return [baseline_correct_predictions, lesk_correct_predictions]

def extract_target_words(instances, key, w):
    word_dict = {}

    count = 0
    for instance_id, wsd_instance in instances.items():
        #print(wsd_instance.lemma)
        if wsd_instance.lemma not in word_dict.keys() and wsd_instance.lemma in w :
            word = wsd_instance.lemma
            synsets = wn.synsets(word)

            if len(synsets) > 1:
                all_synsets_qualify = all(len(synset.examples()) > 0 for synset in synsets)

                if all_synsets_qualify:
                    word_dict[word] = []
                    for synset in synsets:
                        contexts = []
                        name = synset.name()
                        examples = synset.examples()
                        contexts.append(examples[0])
                        word_dict[word].append((name, contexts))
                # If not all synsets qualify, don't add the word to the dictionary
                else:
                    continue
    test_dict = {}

    for instance_id, wsd_instance in instances.items():
        if wsd_instance.lemma in w:
            word = wsd_instance.lemma
            if word not in test_dict.keys():
                test_dict[word] = []
            test_dict[word].append((key[instance_id], wsd_instance.context))
    return word_dict, test_dict


def training_bootstrapping(word_dict, test_dict):
    unlabel_dict = {}
    model_dict = {}
    all_text = []
    acc = 0

    for word in word_dict.keys():
        count = 0
        if word not in unlabel_dict:
            unlabel_dict[word] = []
        for sentence in brown.sents():
            if word in sentence:
                sent = process_text(sentence)
                unlabel_dict[word].append(sent)
                count += 1
                if count >= 50:
                    break
    for word in word_dict:
        for sense, context in word_dict[word]:
            all_text.append(' '.join(context))
        for sentence in unlabel_dict[word]:
            all_text.append(' '.join(sentence))
        for sentences in test_dict[word]:
            for sentence in sentences:
                all_text.append(' '.join(sentence))
    vectorizer = CountVectorizer()
    vectorizer.fit(all_text)

    for word in word_dict.keys():
        training = []
        senses = []
        for sense, context in word_dict[word]:
            senses.append(sense)
            training.append(' '.join(context))
        X_train = vectorizer.transform(training)
        clf = MultinomialNB()
        clf.fit(X_train, senses)
        model_dict[word] = clf

    iteration = 3
    for word in model_dict.keys():
        for i in range(iteration):
            X_test = [' '.join(tokens) for tokens in unlabel_dict[word]]
            X_test = vectorizer.transform(X_test)

            clf = model_dict[word]
            predicted = clf.predict(X_test)
            predicted_probs = clf.predict_proba(X_test)
            high_confidence_threshold = 0.8
            high_confidence_indices = np.where(np.max(predicted_probs, axis=1) > high_confidence_threshold)[0]

            if len(high_confidence_indices) > 0:
                high_confidence_samples = [X_test[i] for i in high_confidence_indices]

                high_confidence_labels = [predicted[i] for i in high_confidence_indices]
                training.extend(high_confidence_samples)
                senses.extend(high_confidence_labels)
                clf.fit(training, senses)
                model_dict[word] = clf
        # pred
        for test_instance in test_dict[word]:
            # Assuming test_instance is a tuple (label, context)
            label, context = test_instance
            X_pred = vectorizer.transform([' '.join(context)])
            clf = model_dict[word]

            predicted = clf.predict(X_pred)
            print(predicted, label)
            if predicted == label:
                acc += 1
    print(acc)
    return model_dict

def training_bootstrapping_2(word_dict, test_dict):
    unlabel_dict = {}
    model_dict = {}
    all_text = []
    acc = 0
    tot = 0

    for word in word_dict.keys():
        count = 0
        if word not in unlabel_dict:
            unlabel_dict[word] = []
        for sentence in brown.sents():
            if word in sentence:
                sent = process_text(sentence)
                unlabel_dict[word].append(sent)
                count += 1
                if count >= 9:
                    break
    for word in word_dict:
        for sense, context in word_dict[word]:
            all_text.append(' '.join(context))
        for sentence in unlabel_dict[word]:
            all_text.append(' '.join(sentence))
        for sentences in test_dict[word]:
            for sentence in sentences:
                all_text.append(' '.join(sentence))

    vectorizer = CountVectorizer()
    vectorizer.fit(all_text)

    for word in word_dict.keys():
        training = []
        senses = []
        local_total = 0
        local_acc = 0

        for sense, context in word_dict[word]:
            senses.append(sense)
            training.append(' '.join(context))
        X_train = vectorizer.transform(training)
        clf = MultinomialNB()
        clf.fit(X_train, senses)

        iteration = 10
        for i in range(iteration):
            print(i)
            X = [' '.join(tokens) for tokens in unlabel_dict[word]]
            X_test = vectorizer.transform(X)

            predicted = clf.predict(X_test)
            predicted_probs = clf.predict_proba(X_test)
            high_confidence_threshold = 0.8

            high_confidence_indices = np.where(np.max(predicted_probs, axis=1) > high_confidence_threshold)[0]

            if len(high_confidence_indices) > 0:
                high_confidence_samples = [X[i] for i in high_confidence_indices]
                high_confidence_labels = [predicted[i] for i in high_confidence_indices]
                X_high_confidence = vectorizer.transform(high_confidence_samples)
                X_train = vstack([X_train, X_high_confidence])

                senses.extend(high_confidence_labels)
                clf.fit(X_train, senses)
                unlabel_dict[word] = [j for j in unlabel_dict[word] if j not in high_confidence_indices]
            else: break
        # pred
        for test_instance in test_dict[word]:
            # Assuming test_instance is a tuple (label, context)
            label, context = test_instance
            X_pred = vectorizer.transform([' '.join(context)])

            predicted = clf.predict(X_pred)
            tot += 1
            local_total += 1
            if predicted == label:
                acc += 1
                local_acc += 1
    return acc / tot

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def get_mean_embedding(output):
    return output.last_hidden_state[:, 1:-1, :].mean(dim=1)
def get_sense_embeddings(word):
    synsets = wn.synsets(word)
    sense_embeddings = {}
    for synset in synsets:
        definition = synset.definition()
        inputs = tokenizer(definition, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = get_mean_embedding(outputs)
        sense_embeddings[synset] = embedding
    return sense_embeddings

def disambiguate_word_in_context(sentence, target_word):
    """Disambiguate a word in the context of a sentence"""
    # Tokenize and get embeddings for the sentence
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embedding = get_mean_embedding(outputs)

    # Get sense embeddings for the target word
    sense_embeddings = get_sense_embeddings(target_word)

    # Find the sense with the highest cosine similarity
    max_similarity = -1
    best_sense = None
    for sense, embedding in sense_embeddings.items():
        similarity = torch.nn.functional.cosine_similarity(sentence_embedding, embedding, dim=1).item()
        if similarity > max_similarity:
            max_similarity = similarity
            best_sense = sense
    return best_sense

if __name__ == '__main__':
    data_f = '/content/sample_data/multilingual-all-words.en.xml'

    key_f = '/content/sample_data/wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)
    w = ['game', 'year', 'player', 'team', 'case']


    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    print(len(dev_instances))  # number of dev instances
    print(len(test_instances))  # number of test instances
    print(len(dev_key))  # number of dev instances
    print(len(test_key))  # number of test instances

    dev_baseline, dev_lesk = evaluate(dev_instances, dev_key)
    correct_baseline, correct_lesk = evaluate(test_instances, test_key)

    accuracy_dev_baseline = dev_baseline / len(dev_instances)
    accuracy_dev_lesk = dev_lesk / len(dev_instances)
    accruacy_baseline = correct_baseline / len(test_instances)
    accruacy_lesk = correct_lesk / len(test_instances)
    print(accuracy_dev_baseline)
    print(accuracy_dev_lesk)
    print(accruacy_baseline)
    print(accruacy_lesk)

    part_test_instances = find_the_instances(test_instances, w)
    part_correct_baseline, part_correct_lesk = evaluate(part_test_instances, test_key)
    part_accruacy_baseline = part_correct_baseline / len(part_test_instances)
    part_accruacy_lesk = part_correct_lesk / len(part_test_instances)

    print(part_accruacy_baseline)
    print(part_accruacy_lesk)

    _, test_dict = extract_target_words(test_instances, test_key, w)
    with open('seed_set_new(1).pkl', 'rb') as f:
        seed_set = pickle.load(f)
    word_dict = {}

    for key in seed_set:
        word_dict[key] = []
        for synset in seed_set[key]:
            word_dict[key].append((synset, seed_set[key][synset]))

    bootstrapping_acc = training_bootstrapping_2(word_dict, test_dict)
    print(bootstrapping_acc)

    acc = 0
    from tqdm import tqdm

    with tqdm(total=len(test_instances)) as pbar:
        for (id, wsd_instance) in test_instances.items():
            pred = disambiguate_word_in_context((' ').join(wsd_instance.context), wsd_instance.lemma)

            print(pred.name(), test_key[id])
            if pred.name() == test_key[id][0]:
                acc += 1
            pbar.update(1)

    pretrain_acc = acc / len(test_instances)

    # Sample data: an array with 4 elements
    data = [accuracy_dev_baseline,accuracy_dev_lesk,accruacy_baseline,accruacy_lesk,pretrain_acc]
    labels = ['baseline for dev_instances','lesk for dev_instances','baseline for test_instances', 'lesk for test_instances', 'pretraining']

    # Creating a horizontal bar plot
    plt.bar(labels, data)

    for index, value in enumerate(data):
        plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')
    # Adding labels and title for clarity
    plt.xlabel('methods')
    plt.ylabel('Accuracy')
    plt.title('Different method predict instances accuracy for WSD')

    # Display the plot
    plt.savefig("acc_plot.png")
    plt.show()

    data_part = [part_accruacy_baseline, part_accruacy_lesk, bootstrapping_acc]
    part_label = ['baseline', "lesk", 'bootstrapping']

    plt.bar(part_label, data_part)
    for index, value in enumerate(data_part):
        plt.text(index, value, f'{value:.4f}', ha='center', va='bottom')
    plt.xlabel('methods')
    plt.ylabel('Accuracy')
    plt.title('Different method predict 5 marked instance accuracy for WSD')

    # Display the plot
    plt.savefig("acc_plot-part.png")
    plt.show()