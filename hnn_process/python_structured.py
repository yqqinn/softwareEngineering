# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize

from nltk import wordpunct_tokenize
from io import StringIO
import inflection

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnlemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

#############################################################################

"""
对于正则表达式的模式字符串，使用原始字符串（raw string）表示，以避免需要对反斜杠进行转义。
"""
PATTERN_VAR_EQUAL = re.compile(r"(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
PATTERN_VAR_FOR = re.compile(r"for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")

def repair_program_io(code):
    """
    使用了更简洁的列表初始化方式 lines_flags = [0] * len(lines)。
    使用 enumerate() 函数来遍历列表并同时获取索引和值。
    """

    # 正则表达式模式，用于第一种情况
    pattern_case1_in = re.compile(r"In ?\[\d+\]: ?")  # 标记1
    pattern_case1_out = re.compile(r"Out ?\[\d+\]: ?")  # 标记2
    pattern_case1_cont = re.compile(r"( )+\.+: ?")  # 标记3

    # 正则表达式模式，用于第二种情况
    pattern_case2_in = re.compile(r">>> ?")  # 标记4
    pattern_case2_cont = re.compile(r"\.\.\. ?")  # 标记5

    patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                pattern_case2_in, pattern_case2_cont]

    lines = code.split("\n")
    lines_flags = [0] * len(lines)
    code_list = []

    # 匹配模式
    for line_idx, line in enumerate(lines):
        for pattern_idx, pattern in enumerate(patterns):
            if re.match(pattern, line):
                lines_flags[line_idx] = pattern_idx + 1
                break
    lines_flags_string = "".join(map(str, lines_flags))

    if lines_flags.count(0) == len(lines_flags):  # 无需修复
        repaired_code = code
        code_list = [code]

    elif re.match(r"(0*1+3*2*0*)+", lines_flags_string) or re.match(r"(0*4+5*0*)+", lines_flags_string):
        repaired_code = ""
        pre_idx = 0
        sub_block = ""
        if lines_flags[0] == 0:
            flag = 0
            while flag == 0:
                repaired_code += lines[pre_idx] + "\n"
                pre_idx += 1
                flag = lines_flags[pre_idx]
            sub_block = repaired_code
            code_list.append(sub_block.strip())
            sub_block = ""

        for idx in range(pre_idx, len(lines_flags)):
            if lines_flags[idx] != 0:
                repaired_line = re.sub(patterns[lines_flags[idx] - 1], "", lines[idx])
                repaired_code += repaired_line + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += repaired_line + "\n"
            else:
                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

        if len(sub_block.strip()):
            code_list.append(sub_block.strip())

    else:  # 非典型情况，仅移除每个 Out 后的 0 标记行
        repaired_code = ""
        sub_block = ""
        bool_after_Out = False
        for idx in range(len(lines_flags)):
            if lines_flags[idx] != 0:
                if lines_flags[idx] == 2:
                    bool_after_Out = True
                else:
                    bool_after_Out = False
                repaired_line = re.sub(patterns[lines_flags[idx] - 1], "", lines[idx])
                repaired_code += repaired_line + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] == 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += repaired_line + "\n"
            else:
                if not bool_after_Out:
                    repaired_code += lines[idx] + "\n"

                if len(sub_block.strip()) and (idx > 0 and lines_flags[idx - 1] != 0):
                    code_list.append(sub_block.strip())
                    sub_block = ""
                sub_block += lines[idx] + "\n"

    return repaired_code, code_list


def get_vars(ast_root):
    return sorted({node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})

def get_vars_heuristics(code):
    varnames = set()
    code_lines = [line for line in code.split("\n") if line.strip()]

    start = 0
    end = len(code_lines) - 1
    success = False
    while not success:
        try:
            root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            success = True

    # 处理剩余部分
    for line in code_lines[end:]:
        line = line.strip()
        try:
            root = ast.parse(line)
        except:
            # 匹配 PATTERN_VAR_EQUAL
            pattern_var_equal_matched = re.match(PATTERN_VAR_EQUAL, line)
            if pattern_var_equal_matched:
                match = pattern_var_equal_matched.group()[:-1]  # 移除 "="
                varnames = varnames.union(set([var.strip() for var in match.split(",")]))

            # 匹配 PATTERN_VAR_FOR
            pattern_var_for_matched = re.search(PATTERN_VAR_FOR, line)
            if pattern_var_for_matched:
                match = pattern_var_for_matched.group()[3:-2]  # remove "for" and "in"
                varnames = varnames.union(set([var.strip() for var in match.split(",")]))

        else:
            varnames = varnames.union(get_vars(root))

    return varnames


def PythonParser(code):
    failed_var = False
    failed_token = False

    try:
        root = ast.parse(code)  # 尝试解析代码
        varnames = set(get_vars(root))  # 获取变量名集合
    except:
        repaired_code, _ = repair_program_io(code)  # 修复代码
        try:
            root = ast.parse(repaired_code)  # 再次尝试解析修复后的代码
            varnames = set(get_vars(root))  # 获取修复后代码的变量名集合
        except:
            failed_var = True  # 变量解析失败
            varnames = get_vars_heuristics(code)

    tokenized_code = []

    def first_trial(_code):
        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(StringIO(_code).readline)
            next(g)
        except:
            return False
        return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)
    g = tokenize.generate_tokens(StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1

        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif token.tok_name[term_type] not in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except:
                failed_token = True  # 标记标记化失败
                code_lines = code.split("\n")
                if lineno <= len(code_lines) - 1:
                    failed_code_line = code_lines[lineno]
                    if posno < len(failed_code_line) - 1:
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(failed_code_line)
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, failed_var, failed_token

#############################################################################

#############################################################################
def revert_abbrev(line):
    """
    使用字典来存储缩略词和对应的完整形式。
    使用re.sub()函数进行替换操作，简化了代码并提高了可读性。
    """
    abbreviations = {
        "it's": "it is",
        "he's": "he is",
        "she's": "she is",
        "that's": "that is",
        "this's": "this is",
        "there's": "there is",
        "here's": "here is"
    }

    line = re.sub(r"\"s", " is", line)
    line = re.sub(r"(?<=[a-zA-Z])\"s", "", line)
    line = re.sub(r"(?<=s)\"s?", "", line)
    line = re.sub(r"(?<=[a-zA-Z])n\"t", " not", line)
    line = re.sub(r"(?<=[a-zA-Z])\"d", " would", line)
    line = re.sub(r"(?<=[a-zA-Z])\"ll", " will", line)
    line = re.sub(r"(?<=[I|i])\"m", " am", line)
    line = re.sub(r"(?<=[a-zA-Z])\"re", " are", line)
    line = re.sub(r"(?<=[a-zA-Z])\"ve", " have", line)

    for abbreviation, full_form in abbreviations.items():
        line = line.replace(abbreviation, full_form)

    return line

def get_wordpos(tag):
    """
    使用了字典映射来获取词性，取代了一系列的if-elif-else语句。
    """
    wordpos_mapping = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }

    return wordpos_mapping.get(tag[0], None)


# ---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    """
    将一些正则表达式的替换操作合并到一个re.sub()调用中，减少了代码行数并提高了可读性。
    将代码中的一些重复操作进行整合，以减少重复的代码段。
    """
    line = revert_abbrev(line)  # 恢复缩略词
    line = re.sub(r'\t+', '\t', line)  # 多个制表符替换为一个制表符
    line = re.sub(r'\n+', '\n', line)  # 多个换行符替换为一个换行符
    line = line.replace('\n', ' ')  # 将换行符替换为空格
    line = re.sub(r' +', ' ', line)  # 多个连续空格替换为一个空格
    line = line.strip()  # 去除首尾空格
    line = inflection.underscore(line)  # 将驼峰命名转换为下划线命名
    line = re.sub(r"\([^\(|^\)]+\)", '', line)  # 去除括号内的内容
    line = line.strip()  # 再次去除首尾空格
    return line


# ---------------------子函数1：句子的分词--------------------
def process_sent_word(line):
    """
    将一些正则表达式的替换操作合并到一个re.sub()调用中，减少了代码行数并提高了可读性。
    将代码中的一些重复操作进行整合，以减少重复的代码段。
    """
    line = re.findall(r"[\w]+|[^\s\w]", line)  # 提取单词和非单词字符
    line = ' '.join(line)  # 单词之间用空格分隔
    line = re.sub(r"\d+(\.\d+)+", 'TAGINT', line)  # 替换小数
    line = re.sub(r'\"[^\"]+\"', 'TAGSTR', line)  # 替换字符串
    line = re.sub(r"0[xX][A-Fa-f0-9]+", 'TAGINT', line)  # 替换十六进制
    line = re.sub(r"\s?\d+\s?", ' TAGINT ', line)  # 替换数字
    line = re.sub(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+", 'TAGOER', line)  # 替换字符
    cut_words = line.split(' ')  # 分割单词
    cut_words = [x.lower() for x in cut_words]  # 全部转为小写
    word_tags = pos_tag(cut_words)  # 词性标注
    tags_dict = dict(word_tags)
    wordnet_lemmatizer = WordNetLemmatizer()
    word_list = []
    for word in cut_words:
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            word = wordnet_lemmatizer.lemmatize(word, pos=word_pos)  # 词性还原
        word = wordnet.morphy(word) if wordnet.morphy(word) else word  # 词干提取
        word_list.append(word)
    return word_list


#############################################################################

def filter_invachar(line, pattern):
    line = re.sub(pattern, ' ', line)  # 去除非常用符号
    line = re.sub('-+', '-', line)  # 中横线
    line = re.sub('_+', '_', line)  # 下划线
    line = line.replace('|', ' ').replace('¦', ' ')  # 去除横杠
    return line

"""
将两个过滤函数filter_all_invachar()和filter_part_invachar()合并为一个通用的filter_invachar()函数，
通过传递不同的正则表达式模式来实现不同的过滤操作。
"""

def filter_all_invachar(line):
    pattern = r'[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+'
    return filter_invachar(line, pattern)

def filter_part_invachar(line):
    pattern = r'[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+'
    return filter_invachar(line, pattern)

########################主函数：代码的tokens#################################
def python_code_parse(line):
    """
    简化了列表生成式中的判断条件，避免使用strip()函数两次。
    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub('>>+', '', line)
    line = re.sub(' +', ' ', line)
    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')
        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        token_list = [x.lower() for x in cut_tokens if x.strip()]
        return token_list
    except:
        return '-1000'
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def python_query_parse(line):
    """
    将去除括号的逻辑整合到列表推导式中，避免使用for循环和修改原始列表。
    使用列表推导式简化了去除空字符串的过程。
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)

    # 去除括号
    word_list = [word for word in word_list if not re.findall('[\(\)]', word)]
    word_list = [word.strip() for word in word_list if word.strip()]

    return word_list


def python_context_parse(line):
    """
    使用列表推导式简化了去除空字符串的过程。
    """
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)

    word_list = [word.strip() for word in word_list if word.strip()]

    return word_list

#######################主函数：句子的tokens##################################

if __name__ == '__main__':
    # 测试 python_query_parse
    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    print(python_query_parse('python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    # 测试 python_context_parse
    print(python_context_parse('How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    # 测试 python_code_parse
    print(python_code_parse('if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    print(python_code_parse("diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    print(python_code_parse('  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
