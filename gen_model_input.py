from bs4 import BeautifulSoup
from model import get_engine, get_session
from model.wos_document import *


def cal_basic_info(termolator_output_file: str):
    """

    **THIS SHOULD BE CALLED BEFORE FURTHER CALCULATING ANY OTHER PROPERTIES**

    Parse and get basic information from the Termolator output,
    including the name, variants, document frequency, term frequency,
    and the source documents of technical terms

    :param termolator_output_file: The term_instance_map file generated by Termolator
    :return: The basic information of each technical terms
    """
    term_dict = {}

    with open(termolator_output_file, mode='r', encoding='utf-8') as term_map:
        single_term = ''
        for line in term_map:
            single_term += line
            if line.startswith('<term '):
                single_term = line
            if line.startswith('</term>'):
                soup = BeautifulSoup(single_term, 'lxml')
                term = soup.find('term')

                term_name = term['string']
                term_rank = term['rank']
                term_tf = term['total_frequency']
                term_df = term['number_of_files_containing_term']
                term_variants = term['variants'].split('|')
                term_docs = set([i['file'].replace('foreground/', '') for i in term.find_all('instance')])

                term_dict[term_name] = {
                    'rank': term_rank,
                    'tf': term_tf,
                    'df': term_df,
                    'variants': term_variants,
                    'docs': term_docs
                }
    return term_dict


def cal_doc_info(term_dict: dict, session):

    for term, info in term_dict.items():
        cited_times = 0

        authors = []
        refs = []
        funds = []
        affs = []
        keywords = []
        keyword_plus = []
        cats = []

        for unique_id in info['docs']:
            doc = session.query(WosDocument).filter(WosDocument.unique_id == unique_id)[0]

            cited_times += doc.cited_times

            single_doc_authors = [(i.first_name.strip() + i.last_name.strip()).lower() for i in doc.authors]
            authors += single_doc_authors

            single_doc_refs = [i.document_md5 for i in doc.references]
            refs += single_doc_refs

            single_doc_funds = [(i.agent + i.funding_number if i.funding_number else '').lower() for i in doc.fundings]
            funds += single_doc_funds

            single_doc_affs = [i.address.lower() for j in doc.authors for i in j.affiliations]
            affs += single_doc_affs

            single_doc_keywords = [i.keyword.lower() for i in doc.keywords]
            keywords += single_doc_keywords

            single_doc_keyword_plus = [i.keyword_plus.lower() for i in doc.keyword_plus]
            keyword_plus += single_doc_keyword_plus

            single_doc_cats = [i.category.lower() for i in doc.categories]
            cats += single_doc_cats

        info['author_num'] = len(set(authors))
        info['ref_num'] = len(set(refs))
        info['aff_num'] = len(set(affs))
        info['kw_num'] = len(set(keywords))
        info['kp_num'] = len(set(keyword_plus))
        info['cat_num'] = len(set(cats))
        info['cited_times'] = cited_times

        print(term, info)
        break


if __name__ == '__main__':
    engine = get_engine(db_url='sqlite:///../data/gene_editing.db')
    session = get_session(engine)

    term_dict = cal_basic_info(r'..\data\Termolator_result\gene_editing\gene.term_instance_map')
    cal_doc_info(term_dict, session)

    session.close()