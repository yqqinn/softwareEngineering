
'''
并行分词
'''

import os
import pickle
import logging
import sys
sys.path.append("..")

# 解析结构
from python_structured import *
from sqlang_structured import *

# FastText库  gensim 3.4.0
from gensim.models import FastText

import numpy as np

# 词频统计库
import collections
# 词云展示库
import wordcloud
# 图像处理库 Pillow 5.1.0
from PIL import Image

# 多进程
from multiprocessing import Pool as ThreadPool



#python解析
def multipro_python_query(data_list):
    result = [python_query_parse(line) for line in data_list]
    return result

def multipro_python_code(data_list):
    result = [python_code_parse(line) for line in data_list]
    return result

def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


#sql解析
def multipro_sqlang_query(data_list):
    result = [sqlang_query_parse(line) for line in data_list]
    return result

def multipro_sqlang_code(data_list):
    result = [sqlang_code_parse(line) for line in data_list]
    return result

def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result


def parse_python(python_list, split_num):
    # 解析acont1
    acont1_data = [i[1][0][0] for i in python_list]
    acont1_cut = parse_data_multiprocess(acont1_data, split_num, multipro_python_context)
    print('acont1条数：%d' % len(acont1_cut))

    # 解析acont2
    acont2_data = [i[1][1][0] for i in python_list]
    acont2_cut = parse_data_multiprocess(acont2_data, split_num, multipro_python_context)
    print('acont2条数：%d' % len(acont2_cut))

    # 解析query
    query_data = [i[3][0] for i in python_list]
    query_cut = parse_data_multiprocess(query_data, split_num, multipro_python_query)
    print('query条数：%d' % len(query_cut))

    # 解析code
    code_data = [i[2][0][0] for i in python_list]
    code_cut = parse_data_multiprocess(code_data, split_num, multipro_python_code)
    print('code条数：%d' % len(code_cut))

    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))

    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def parse_data_multiprocess(data, split_num, parse_function):
    split_data = [data[i:i + split_num] for i in range(0, len(data), split_num)]
    pool = ThreadPool(10)
    parsed_data_list = pool.map(parse_function, split_data)
    pool.close()
    pool.join()
    parsed_data = []
    for p in parsed_data_list:
        parsed_data += p
    return parsed_data


def parse_sqlang(sqlang_list, split_num):
    # 解析acont1
    acont1_data = [i[1][0][0] for i in sqlang_list]
    acont1_cut = parse_data_multiprocess(acont1_data, split_num, multipro_sqlang_context)
    print('acont1条数：%d' % len(acont1_cut))

    # 解析acont2
    acont2_data = [i[1][1][0] for i in sqlang_list]
    acont2_cut = parse_data_multiprocess(acont2_data, split_num, multipro_sqlang_context)
    print('acont2条数：%d' % len(acont2_cut))

    # 解析query
    query_data = [i[3][0] for i in sqlang_list]
    query_cut = parse_data_multiprocess(query_data, split_num, multipro_sqlang_query)
    print('query条数：%d' % len(query_cut))

    # 解析code
    code_data = [i[2][0][0] for i in sqlang_list]
    code_cut = parse_data_multiprocess(code_data, split_num, multipro_sqlang_code)
    print('code条数：%d' % len(code_cut))

    qids = [i[0] for i in sqlang_list]

    return acont1_cut, acont2_cut, query_cut, code_cut, qids


def parse_data_multiprocess(data, split_num, parse_function):
    split_data = [data[i:i + split_num] for i in range(0, len(data), split_num)]
    pool = ThreadPool(10)
    parsed_data_list = pool.map(parse_function, split_data)
    pool.close()
    pool.join()
    parsed_data = []
    for p in parsed_data_list:
        parsed_data += p
    return parsed_data


def main(lang_type, split_num, source_path, save_path):
    total_data = []

    with open(source_path, "rb") as f:
        corpus_list = pickle.load(f)

        if lang_type == 'python':
            parse_acont1, parse_acont2, parse_query, parse_code, qids = parse_python(corpus_list, split_num)
        elif lang_type == 'sql':
            parse_acont1, parse_acont2, parse_query, parse_code, qids = parse_sqlang(corpus_list, split_num)

        for i in range(len(qids)):
            total_data.append([qids[i], [parse_acont1[i], parse_acont2[i]], [parse_code[i]], parse_query[i]])

    with open(save_path, "w") as f:
        f.write(str(total_data))


python_type= 'python'
sqlang_type ='sql'
words_top = 100
split_num = 1000


def test(path1, path2):
    with open(path1, "rb") as f:
        corpus_lis1 = pickle.load(f)
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read())

    print(corpus_lis1[10])
    print(corpus_lis2[10])

if __name__ == '__main__':
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save ='../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'

    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'

    #main(sqlang_type,split_num,staqc_sql_path,staqc_sql_save)
    #main(python_type, split_num, staqc_python_path, staqc_python_save)

    large_python_path='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'

    large_sql_path='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save='../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'

   #main(sqlang_type, split_num, large_sql_path, large_sql_save)
    main(python_type, split_num, large_python_path, large_python_save)