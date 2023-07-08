
'''
从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

import numpy as np
import pickle
from gensim.models import KeyedVectors

def trans_bin(path1,path2):
    """
    将词向量文件保存成二进制格式（.bin）
    """
    # 加载词向量文件（非二进制格式）
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 进行必要的预处理和转换
    wv_from_text.init_sims(replace=True)
    # 保存转换后的词向量为二进制文件
    wv_from_text.save(path2)


def get_new_dict(type_vec_path,type_word_path,final_vec_path,final_word_path):
    """
    从大词典中获取特定于语料的词典，并构建词向量矩阵
    """
    # 加载转换文件
    model = KeyedVectors.load(type_vec_path, mmap='r')

    # 读取包含特定语料词的文件并转换为列表形式
    with open(type_word_path,'r')as f:
        total_word= eval(f.read())
        f.close()

    #通过使用字典来存储特殊标记的嵌入向量，使得代码更加简洁和易读
    rng = np.random.RandomState(None)
    special_tokens = {'PAD': np.zeros(shape=(1, 300)).squeeze(),
                      'UNK': rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze(),
                      'SOS': rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze(),
                      'EOS': rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
                      }

    word_dict = list(special_tokens.keys())

    # 存储无法找到词向量的词
    fail_word = []
    word_vectors = [special_tokens[token] for token in word_dict]
    print(len(total_word))


    for word in total_word:
        try:
            # 加载词向量并将其添加到嵌入向量列表和词典中
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            # 如果无法找到词向量，则将该词添加到失败词列表中
            print(word)
            fail_word.append(word)

    # 打印词典、嵌入向量和失败词的数量
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))


    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)

    # 保存词向量和词典为文件
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    # 重新加载保存的词向量和词典
    v = pickle.load(open(final_vec_path, 'rb'), encoding='iso-8859-1')
    with open(final_word_path, 'rb') as f:
        word_dict = pickle.load(f)

    count = 0
    for i in range(4, len(word_dict)):
        # 检查嵌入向量是否正确加载
        if word_vectors[i].all() == model.wv[word_dict[i]].all():
            continue
        else:
            count += 1

    # 打印错误向量的数量
    print(count)

    print("完成")



def get_index(type, text, word_dict):
    """
    得到词在词典中的位置：对于类型为'code'的情况，只需要找到文本中词在词典中的位置，并在前后添加标记索引1和2。
    对于其他类型的文本，只需要找到每个词在词典中的位置，并在开头添加索引0（表示空白文本或未知词）。
    """
    location = []
    if type == 'code':
        location.append(1)
        max_len = min(348, len(text))  # 限制最大长度为348
        for i in range(max_len):
            token = text[i]
            index = word_dict.get(token, word_dict['UNK'])  # 获取词在词典中的位置，如果不存在则返回UNK的位置
            location.append(index)
        location.append(2)
    else:
        for token in text:
            index = word_dict.get(token, word_dict['UNK'])  # 获取词在词典中的位置，如果不存在则返回UNK的位置
            location.append(index)
        if len(location) == 0 or location[0] == word_dict['UNK']:
            location.insert(0, 0)  # 在开头插入0表示空白文本或未知词

    return location



def Serialization(word_dict_path, type_path, final_type_path):
    """
    将训练、测试或验证语料进行序列化处理：使用更简洁的方式处理列表的填充和切片操作。
    通过直接对列表进行切片，并使用列表相加的方式，可以避免使用显式的循环和条件语句。
    同时，将读取文件和写入文件的代码放在了适当的位置，提高了代码的可读性。
    """
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for item in corpus:
        qid = item[0]  # 查询ID

        # 处理Si的词列表
        Si_word_list = get_index('text', item[1][0], word_dict)
        Si_word_list = Si_word_list[:100] + [0] * max(0, 100 - len(Si_word_list))  # 填充到长度为100

        # 处理Si+1的词列表
        Si1_word_list = get_index('text', item[1][1], word_dict)
        Si1_word_list = Si1_word_list[:100] + [0] * max(0, 100 - len(Si1_word_list))  # 填充到长度为100

        # 处理代码的词列表
        tokenized_code = get_index('code', item[2][0], word_dict)
        tokenized_code = tokenized_code[:350] + [0] * max(0, 350 - len(tokenized_code))  # 填充到长度为350

        # 处理查询的词列表
        query_word_list = get_index('text', item[3], word_dict)
        query_word_list = query_word_list[:25] + [0] * max(0, 25 - len(query_word_list))  # 填充到长度为25

        block_length = 4  # 代码块长度
        label = 0  # 标签

        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)



def get_new_dict_append(type_vec_path,previous_dict,previous_vec,append_word_path,final_vec_path,final_word_path):  #词标签，词向量

    # 加载转换文件

    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    with open(append_word_path,'r')as f:
        append_word= eval(f.read())
        f.close()

    # 输出词向量
    print(type(pre_word_vec))
    word_dict =  list(pre_word_dict.keys()) # 先将原有的词典转换为列表
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word =[]

    rng = np.random.RandomState(None)
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze() # 使用 squeeze() 简化代码

    for word in append_word:
        try:
            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            fail_word.append(word)

    # 输出更新后的词典信息
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])

    # 保存更新后的词向量和词典
    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))


    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")


import time

#-------------------------参数配置----------------------------------
#python 词典 ：1121543 300


if __name__ == '__main__':
    """
    将各个路径和文件名的定义提前，以便更清晰地查看和修改。
    添加了适当的空行和注释，提高代码的可读性。
    调整了代码的缩进，使其更符合 Python 的惯例。
    删除了一些不必要的空格和冗余代码。
    """
    # Python Struc2Vec 文件路径
    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt' #239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin' #2s

    # SQL Struc2Vec 文件路径
    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # 最初基于 Staqc 的词典和词向量路径
    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # 最后打标签的语料路径
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # 加载 Python Struc2Vec 文件并转换为二进制格式
    # trans_bin(ps_path, ps_path_bin)

    # 加载 SQL Struc2Vec 文件并转换为二进制格式
    # trans_bin(sql_path, sql_path_bin)

    # 获取最初基于 Staqc 的词典和词向量
    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # 获取最后打标签的 SQL 语料的词典和词向量
    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)

    # 序列化 SQL 语料
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # 获取最后打标签的 Python 语料的词典和词向量
    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)

    # 序列化 Python 语料
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    # Serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    #test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)








