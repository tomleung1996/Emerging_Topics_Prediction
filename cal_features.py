from model import get_engine, get_session
from model.wos_document import *
from collections import defaultdict
import pickle
from tqdm import tqdm

if __name__ == '__main__':
    term_dict = {}
    freq_dict = {}

    with open(r'data/gene_editing.txt', 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            term_dict[i] = line.strip().split('\t')
            freq_dict[i] = defaultdict(int)

    engine = get_engine(db_url='sqlite:///data/gene_editing.db')
    session = get_session(engine)

    # print(freq_dicts)

    for wos_document in tqdm(session.query(WosDocument)):
        if not wos_document.pub_year:
            continue

        title = wos_document.title + '.\n'
        abs = wos_document.abs.replace('. ', '.\n') if wos_document.abs else ''
        ks = '. '.join([k.keyword for k in wos_document.keywords]) + '.\n' if len(wos_document.keywords) > 0 else ''
        kps = '. '.join([k.keyword_plus for k in wos_document.keyword_plus]) + '.\n' if len(
            wos_document.keyword_plus) > 0 else ''

        concat = (title + ks + kps + abs).lower()

        for i, term in term_dict.items():
            for t in term:
                if len(t) < 5:
                    continue
                print(t)
                if t in concat:
                    print(t, wos_document.unique_id)
                    freq_dict[i][int(wos_document.pub_year)] += 1
                    break

    pickle.dump(freq_dict, open(r'output/freq_dicts.list', 'wb'))

    session.close()