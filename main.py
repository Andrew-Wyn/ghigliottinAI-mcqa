import json
#import gemini_generate
import random
from tqdm import tqdm
import time
import fasttext.util
import nltk
from nltk.corpus import wordnet as wn
from numpy import dot
from numpy.linalg import norm


nltk.download('wordnet')
nltk.download('omw-1.4')

DEV_SET_PATH = ""
OUTPUT_PATH = ""
LIST_WORDS = random.sample([" ".join(x.split("_")).lower() for x in list(wn.all_lemma_names(lang="ita")) if (len(x) >= 3 and len(x) <= 10 and not "_" in x and not " " in x)], 2500)
LIST_WORDS_EMBEDDINGS = {}

#exit()

fasttext.util.download_model('it', if_exists='ignore')
ft = fasttext.load_model('cc.it.300.bin')


def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


def build_ft_embeddings():
    for w in LIST_WORDS:
        LIST_WORDS_EMBEDDINGS[w] = ft.get_word_vector(w)


def load_dev_set():
    # load json dataset
    with open(DEV_SET_PATH, "r") as f:
        dev_set = json.load(f)

    return dev_set


def get_not_hint(game_words):
    while True:
        not_hint = random.choice(LIST_WORDS)

        if not not_hint in game_words:
            break

    return not_hint

"""
def generate_distractors(w1, w2, w3, w4, w5, solution):
    not_hint_1 = get_not_hint([w1, w2, w3, w4, w5, solution])
    not_hint_2 = get_not_hint([w1, w2, w3, w4, w5, solution, not_hint_1])

    return gemini_generate.generate_distractors(w1, w2, w3, w4, w5, solution, not_hint_1, not_hint_2)
"""

def generate_distractors_fasttext(w1, w2, w3, w4, w5, solution):
    # create the embedding of each word of the game

    w1_emb = ft.get_word_vector(w1)
    w2_emb = ft.get_word_vector(w2)
    w3_emb = ft.get_word_vector(w3)
    w4_emb = ft.get_word_vector(w4)
    w5_emb = ft.get_word_vector(w5)
    solution_emb = ft.get_word_vector(solution)

    game_scored_words = LIST_WORDS_EMBEDDINGS.copy() # copy to have 

    for word, value in game_scored_words.items():
        random_elements = random.sample(range(5), 3)

        element_to_sum = [cos_sim(w1_emb, value), cos_sim(w2_emb, value), cos_sim(w3_emb, value), cos_sim(w4_emb, value), cos_sim(w5_emb, value)]

        game_scored_words[word] = 0
        
        for i in range(5):
            if i in random_elements:
                game_scored_words[word] -= element_to_sum[i]
            else:
                game_scored_words[word] += element_to_sum[i]
        
        game_scored_words[word] -= cos_sim(solution_emb, value)

    # sort game_scored_words reverse getting only words
    words_sorted = [k for k, _ in sorted(game_scored_words.items(), key=lambda item: item[1], reverse=False)]

    if w1 in words_sorted:
        words_sorted.remove(w1)
    if w2 in words_sorted:
        words_sorted.remove(w2)
    if w3 in words_sorted:
        words_sorted.remove(w3)
    if w4 in words_sorted:
        words_sorted.remove(w4)
    if w5 in words_sorted:
        words_sorted.remove(w5)
    if solution in words_sorted:
        words_sorted.remove(solution)

    # get words that have length similar to the solution

    if len(solution) <= 10 and len(solution) >= 3:
        words_sorted = [w for w in words_sorted if (len(w) <= len(solution) + 1 and len(w) >= len(solution) - 1)]
    else:    
        words_sorted = [w for w in words_sorted]

    del game_scored_words

    #print(w1, w2, w3, w4, w5, solution)
    #print(words_sorted[5:8])

    return words_sorted[3:6]

def generate_distractors_naive(w1, w2, w3, w4, w5, solution):
    not_hint_1 = get_not_hint([w1, w2, w3, w4, w5, solution])
    not_hint_2 = get_not_hint([w1, w2, w3, w4, w5, solution, not_hint_1])
    not_hint_3 = get_not_hint([w1, w2, w3, w4, w5, solution, not_hint_1, not_hint_2])

    return [not_hint_1, not_hint_2, not_hint_3]


def expand_game(game):
    w1, w2, w3, w4, w5, solution = game["w1"], game["w2"], game["w3"], game["w4"], game["w5"], game["solution"]
    distractors = generate_distractors_fasttext(w1, w2, w3, w4, w5, solution)
    ordered_choices = distractors + [solution]

    indexes = list(range(4))
    random.shuffle(indexes)

    choices = [ordered_choices[i] for i in indexes]
    label = indexes.index(3)

    return {
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w4": w4,
        "w5": w5,
        "choices": choices,
        "label": label
    }


def expand_dataset(dev_set):
    expanded_dataset = []

    for game in tqdm(dev_set):
        while True:
            try:
                expanded_game = expand_game(game)
                break
            except Exception as e:
                print("Errore:", e)
                # wait some time to admit free usage
                time.sleep(10) # sleep 5 seconds

        expanded_dataset.append(expanded_game)

    return expanded_dataset


def save_dataset(dataset, test_split):

    random.shuffle(dataset)

    split_int = int(len(dataset)*test_split)

    train_dataset, test_dataset = dataset[split_int:], dataset[:split_int]

    with open(OUTPUT_PATH+"/train.jsonl", "w") as f:
        for el in train_dataset:
            f.writelines(json.dumps(el) + "\n")

    with open(OUTPUT_PATH+"/test.jsonl", "w") as f:
        for el in test_dataset:
            f.writelines(json.dumps(el) + "\n")


def main():
    # Load Dataset
    dev_set = load_dev_set()

    # Build words embeddings
    build_ft_embeddings()

    # Configure Model
    #gemini_generate.configure_model()
    
    # Expand Dataset
    expanded_dataset = expand_dataset(dev_set)

    # Save Dataset
    save_dataset(expanded_dataset, test_split=0.9)


if __name__ == "__main__":
    main()