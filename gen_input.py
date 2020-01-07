from model import get_engine, get_session
from model.wos_document import *
import glob


def output_text(db_url: str, output_path: str):
    engine = get_engine(db_url=db_url)
    session = get_session(engine)

    cache = []
    for wos_document in session.query(WosDocument):
        uid = wos_document.unique_id
        title = wos_document.title + '.\n'
        abs = wos_document.abs.replace('. ', '.\n') if wos_document.abs else ''
        ks = '. '.join([k.keyword for k in wos_document.keywords]) + '.\n' if len(wos_document.keywords) > 0 else ''
        kps = '. '.join([k.keyword_plus for k in wos_document.keyword_plus]) + '.\n' if len(
            wos_document.keyword_plus) > 0 else ''

        cache.append(title + ks + kps + abs + '\n\n')

        if len(cache) >= 30:
            with open(r'{}/{}.txt'.format(output_path, uid), mode='w', encoding='utf-8') as file:
                file.writelines(cache)
                cache.clear()

    if len(cache) > 0:
        with open(r'{}/last.txt'.format(output_path), mode='w', encoding='utf-8') as file:
            file.writelines(cache)
            cache.clear()

    session.close()


def output_list(file_path: str, list_path: str):
    with open(list_path, mode='w', encoding='utf-8') as file:
        prefix = '/home/tomleung/The_Termolator-master/gene_editing/'
        for i in glob.glob(file_path):
            file.write(prefix + i.replace('E:\\TomLeung\\Emerging Topics\\output\\', '').replace('\\', '/') + '\n')


if __name__ == '__main__':
    output_list(r'E:\TomLeung\Emerging Topics\output\background\*.txt', r'E:\TomLeung\Emerging Topics\output\background.list')
