import pickle

if __name__ == '__main__':
    freq_dicts = pickle.load(open(r'output/freq_dicts.list', 'rb'))

    term_dict = {}

    with open(r'data/gene_editing.txt', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            term_dict[i] = line.strip().split('\t')


    print(freq_dicts[3])