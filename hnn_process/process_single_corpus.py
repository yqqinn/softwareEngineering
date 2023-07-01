import pickle
from collections import Counter

def load_pickle(filename):
    return pickle.load(open(filename, 'rb'), encoding='iso-8859-1')

def single_list(arr, target):
    return arr.count(target)

def data_staqc_processing(filepath,save_single_path,save_multiple_path):
    """
    修改了函数名data_staqc_prpcessing为data_staqc_processing，以修正拼写错误。
    修改了save_mutiple_path为save_multiple_path。
    使用列表推导式来生成total_data_single和total_data_multiple，使代码更简洁。
    使用with open语句来代替手动打开和关闭文件，确保文件操作的正确性和安全性。
    """
    with open(filepath,'r')as f:
        total_data= eval(f.read())

    # 提取所有问题ID（qids）
    qids = [data[0][0] for data in total_data]
    # 使用Counter统计每个问题ID出现的次数，用于判断是否为单候选或多候选
    result = Counter(qids)

    # 将单候选问题和多候选问题分别保存到不同的列表中
    total_data_single = [data for data in total_data if result[data[0][0]] == 1]
    total_data_multiple = [data for data in total_data if result[data[0][0]] > 1]

    # 将单候选问题数据保存到文件
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))

    # 将多候选问题数据保存到文件
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))


def data_large_processing(filepath,save_single_path,save_multiple_path):
    """
    修改了函数名data_large_prpcessing为data_large_processing，以修正拼写错误。
    修改了save_mutiple_path为save_multiple_path。
    使用列表推导式简化了对total_data中的元素进行提取和筛选的过程。
    使用with open语句替代了open和close语句，以确保文件在使用后被正确关闭。
    """
    # 从文件中加载数据
    total_data = load_pickle(filepath)

    # 提取所有问题ID（qids）
    qids = [data[0][0] for data in total_data]
    result = Counter(qids)

    # 将单候选问题和多候选问题分别保存到不同的列表中
    total_data_single = [data for data in total_data if result[data[0][0]] == 1]
    total_data_multiple = [data for data in total_data if result[data[0][0]] > 1]

    # 将单候选问题数据保存到文件
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)

    # 将多候选问题数据保存到文件
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)

def single_unlable2lable(path1,path2):
    """
    优化了对有标签数据的生成过程，使用列表推导式进行简化。
    """
    total_data = load_pickle(path1)
    labels = [[data[0], 1] for data in total_data]

    # 按照问题ID和标签进行排序
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))

    # 将有标签的数据保存到文件
    with open(path2, 'w') as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    #将staqc_python中的单候选和多候选分开
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_sigle_save ='../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_python_path,staqc_python_sigle_save,staqc_python_multiple_save)

    #将staqc_sql中的单候选和多候选分开
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_sigle_save = '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_sql_path, staqc_sql_sigle_save, staqc_sql_multiple_save)

    #将large_python中的单候选和多候选分开
    large_python_path = '../hnn_process/ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    # 将large_sql中的单候选和多候选分开
    large_sql_path = '../hnn_process/ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    #data_large_prpcessing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    #single_unlable2lable(large_sql_single_save,large_sql_single_label_save)
    #single_unlable2lable(large_python_single_save, large_python_single_label_save)
