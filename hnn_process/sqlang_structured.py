# -*- coding: utf-8 -*-
import re
import sqlparse #0.4.2
import inflection
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnlemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet

#############################################################################
OTHER = 0
FUNCTION = 1
BLANK = 2
KEYWORD = 3
INTERNAL = 4
TABLE = 5
COLUMN = 6
INTEGER = 7
FLOAT = 8
HEX = 9
STRING = 10
WILDCARD = 11
SUBQUERY = 12
DUD = 13

# 定义类型名称字典
ttypes = {
    0: "OTHER",
    1: "FUNCTION",
    2: "BLANK",
    3: "KEYWORD",
    4: "INTERNAL",
    5: "TABLE",
    6: "COLUMN",
    7: "INTEGER",
    8: "FLOAT",
    9: "HEX",
    10: "STRING",
    11: "WILDCARD",
    12: "SUBQUERY",
    13: "DUD",
}

# 定义扫描器规则
scanner = re.Scanner([
    (r"\[[^\]]*\]", lambda scanner, token: token),
    (r"\+", lambda scanner, token: "REGPLU"),
    (r"\*", lambda scanner, token: "REGAST"),
    (r"%", lambda scanner, token: "REGCOL"),
    (r"\^", lambda scanner, token: "REGSTA"),
    (r"\$", lambda scanner, token: "REGEND"),
    (r"\?", lambda scanner, token: "REGQUE"),
    (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
    (r'.', lambda scanner, token: None),
])

#---------------------子函数1：代码的规则--------------------
def tokenizeRegex(s):
    results = scanner.scan(s)[0]
    return results

#---------------------子函数2：代码的规则--------------------
class SqlangParser():
    def __init__(self, sql, regex=False, rename=True):
        self.sql = self.sanitizeSql(sql)
        self.idMap = {"COLUMN": {}, "TABLE": {}}
        self.idMapInv = {}
        self.idCount = {"COLUMN": 0, "TABLE": 0}
        self.regex = regex
        self.parse = sqlparse.parse(self.sql)
        self.parse = [self.parse[0]]
        self.removeWhitespaces(self.parse[0])
        self.identifyLiterals(self.parse[0])
        self.parse[0].ptype = SUBQUERY
        self.identifySubQueries(self.parse[0])
        self.identifyFunctions(self.parse[0])
        self.identifyTables(self.parse[0])
        self.parseStrings(self.parse[0])

        if rename:
            self.renameIdentifiers(self.parse[0])

        self.tokens = self.getTokens(self.parse)

    @staticmethod
    def sanitizeSql(sql):
        s = sql.strip().lower()
        if not s[-1] == ";":
            s += ';'
        s = re.sub(r'\(', r' ( ', s)
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        s = s.replace('#', '')
        return s

    @staticmethod
    def getTokens(parse):
        flatParse = []
        for expr in parse:
            for token in expr.flatten():
                if token.ttype == STRING:
                    flatParse.extend(str(token).split(' '))
                else:
                    flatParse.append(str(token))
        return flatParse

    def removeWhitespaces(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            tmpChildren = []
            for c in tok.tokens:
                if not c.is_whitespace:
                    tmpChildren.append(c)
            tok.tokens = tmpChildren
            for c in tok.tokens:
                self.removeWhitespaces(c)

    def identifySubQueries(self, tokenList):
        isSubQuery = False
        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                subQuery = self.identifySubQueries(tok)
                if subQuery and isinstance(tok, sqlparse.sql.Parenthesis):
                    tok.ttype = SUBQUERY
            elif str(tok) == "select":
                isSubQuery = True
        return isSubQuery

    def identifyLiterals(self, tokenList):
        blankTokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
        blankTokenTypes = [sqlparse.sql.Identifier]

        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.TokenList):
                tok.ptype = INTERNAL
                self.identifyLiterals(tok)
            elif tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select":
                tok.ttype = KEYWORD
            elif tok.ttype in [sqlparse.tokens.Number.Integer, sqlparse.tokens.Literal.Number.Integer]:
                tok.ttype = INTEGER
            elif tok.ttype in [sqlparse.tokens.Number.Hexadecimal, sqlparse.tokens.Literal.Number.Hexadecimal]:
                tok.ttype = HEX
            elif tok.ttype in [sqlparse.tokens.Number.Float, sqlparse.tokens.Literal.Number.Float]:
                tok.ttype = FLOAT
            elif tok.ttype in [sqlparse.tokens.String.Symbol, sqlparse.tokens.String.Single, sqlparse.tokens.Literal.String.Single, sqlparse.tokens.Literal.String.Symbol]:
                tok.ttype = STRING
            elif tok.ttype == sqlparse.tokens.Wildcard:
                tok.ttype = WILDCARD
            elif tok.ttype in blankTokens or type(tok) in blankTokenTypes:
                tok.ttype = COLUMN

    def identifyFunctions(self, tokenList):
        for tok in tokenList.tokens:
            if isinstance(tok, sqlparse.sql.Function):
                self.parseTreeSentinel = True
            elif isinstance(tok, sqlparse.sql.Parenthesis):
                self.parseTreeSentinel = False
            if self.parseTreeSentinel:
                tok.ttype = FUNCTION
            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyFunctions(tok)

    def identifyTables(self, tokenList):
        if tokenList.ptype == SUBQUERY:
            self.tableStack.append(False)

        for i in range(len(tokenList.tokens)):
            prevtok = tokenList.tokens[i - 1]
            tok = tokenList.tokens[i]

            if str(tok) == "." and tok.ttype == sqlparse.tokens.Punctuation and prevtok.ttype == COLUMN:
                prevtok.ttype = TABLE
            elif str(tok) == "from" and tok.ttype == sqlparse.tokens.Keyword:
                self.tableStack[-1] = True
            elif (str(tok) == "where" or str(tok) == "on" or str(tok) == "group" or str(tok) == "order" or str(tok) == "union") and tok.ttype == sqlparse.tokens.Keyword:
                self.tableStack[-1] = False

            if isinstance(tok, sqlparse.sql.TokenList):
                self.identifyTables(tok)
            elif tok.ttype == COLUMN:
                if self.tableStack[-1]:
                    tok.ttype = TABLE

        if tokenList.ptype == SUBQUERY:
            self.tableStack.pop()

    def parseStrings(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"

    def renameIdentifiers(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameIdentifiers(c)
        elif tok.ttype == COLUMN:
            if str(tok) not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][str(tok)] = colname
                self.idMapInv[colname] = str(tok)
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][str(tok)]
        elif tok.ttype == TABLE:
            if str(tok) not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][str(tok)] = tabname
                self.idMapInv[tabname] = str(tok)
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][str(tok)]
        elif tok.ttype == FLOAT:
            tok.value = "CODFLO"
        elif tok.ttype == INTEGER:
            tok.value = "CODINT"
        elif tok.ttype == HEX:
            tok.value = "CODHEX"

    def parseSql(self):
        return [str(tok) for tok in self.tokens]

    def __str__(self):
        return ' '.join([str(tok) for tok in self.tokens])

    def __hash__(self):
        return hash(tuple([str(x) for x in self.tokensWithBlanks]))
#############################################################################

#############################################################################
"""
将正则表达式模式和替换字符串放入一个列表中，以避免重复的代码，并使代码更易于管理和扩展。
使用更具描述性的变量名 patterns 替代原来的 pat_* 变量，使其更易于理解。
将词性标注的字典定义为 tag_dict，并使用 get() 方法获取词性，避免使用多个 if-elif 语句，使代码更简洁。
"""

# 需要用到的正则表达式模式
patterns = [
    (re.compile("(it|he|she|that|this|there|here)(\"s)", re.I), r"\1 is"),  # 's
    (re.compile("(?<=[a-zA-Z])\"s"), ""),  # s
    (re.compile("(?<=s)\"s?"), ""),  # s
    (re.compile("(?<=[a-zA-Z])n\"t"), " not"),  # not
    (re.compile("(?<=[a-zA-Z])\"d"), " would"),  # would
    (re.compile("(?<=[a-zA-Z])\"ll"), " will"),  # will
    (re.compile("(?<=[I|i])\"m"), " am"),  # am
    (re.compile("(?<=[a-zA-Z])\"re"), " are"),  # are
    (re.compile("(?<=[a-zA-Z])\"ve"), " have")  # have
]

def revert_abbrev(line):
    for pattern, replacement in patterns:
        line = pattern.sub(replacement, line)
    return line

def get_wordpos(tag):
    tag_dict = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag[0], None)

#---------------------子函数1：句子的去冗--------------------
def process_nl_line(line):
    """
    使用原生的正则表达式模式 r'' 代替字符串中的转义字符 \。
    使用 re.sub 一次性处理多个连续的制表符和换行符，避免多次调用 re.sub。
    使用 line.strip() 一次性去除字符串两端的空格。
    将正则表达式模式和替换字符串的定义直接放在 re.sub 中，避免了使用额外的变量。
    使用 rstrip('.') 去除末尾的点，避免了使用额外的正则表达式。
    """
    # 句子预处理
    line = revert_abbrev(line)
    line = re.sub(r'[\t\n]+', ' ', line)
    line = line.strip()

    # 骆驼命名转下划线
    line = inflection.underscore(line)

    # 去除括号里内容
    line = re.sub(r'\([^()]+\)', '', line)

    # 去除末尾的点和空格
    line = line.rstrip('.')

    return line


#---------------------子函数1：句子的分词--------------------
def process_sent_word(line):
    """
    将替换操作的正则表达式模式和替换字符串直接放在 re.sub 中，避免使用额外的变量。
    使用列表推导式将切割后的单词列表中的单词全部转换为小写。
    在循环中处理特殊标记的单词（如 TAGINT、TAGSTR、TAGOER）。
    将词性标注、词性还原和词干提取的逻辑放在循环内部，以提高代码的可读性和简洁性。
    """
    # 找单词
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    # 替换小数
    decimal = re.compile(r"\d+(\.\d+)+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换字符串
    string = re.compile(r'\"[^\"]+\"')
    line = re.sub(string, 'TAGSTR', line)
    # 替换十六进制
    decimal = re.compile(r"0[xX][A-Fa-f0-9]+")
    line = re.sub(decimal, 'TAGINT', line)
    # 替换数字 56
    number = re.compile(r"\s?\d+\s?")
    line = re.sub(number, ' TAGINT ', line)
    # 替换字符 6c60b8e1
    other = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")  # 后缀匹配
    line = re.sub(other, 'TAGOER', line)

    cut_words = line.split(' ')
    # 全部小写化
    cut_words = [x.lower() for x in cut_words]
    word_tags = pos_tag(cut_words)
    tags_dict = dict(word_tags)
    word_list = []
    for word in cut_words:
        if word in ['TAGINT', 'TAGSTR', 'TAGOER']:
            word_list.append(word)
        else:
            word_pos = get_wordpos(tags_dict[word])
            if word_pos in ['a', 'v', 'n', 'r']:
                # 词性还原
                word = wnlemmatizer.lemmatize(word, pos=word_pos)
            # 词干提取(效果最好）
            word = wordnet.morphy(word) if wordnet.morphy(word) else word
            word_list.append(word)

    return word_list

#############################################################################

def filter_all_invachar(line):
    # 去除非常用符号；防止解析有误
    line = re.sub('[^0-9a-zA-Z_\'\"()\n-]', ' ', line)
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def filter_part_invachar(line):
    # 去除非常用符号；防止解析有误
    line= re.sub('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+',' ', line)
    # 中横线
    line = re.sub('-+', '-', line)
    # 下划线
    line = re.sub('_+', '_', line)
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

########################主函数：代码的tokens#################################
def sqlang_code_parse(line):
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)

    line = re.sub('>>+', '', line)  # 新增：去除连续的多个 > 符号
    line = re.sub(r"\d+(\.\d+)+", 'number', line)  # 新增：替换小数

    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        query = SqlangParser(line, regex=True)
        typedCode = query.parseSql()
        typedCode = typedCode[:-1]
        # 骆驼命名转下划线
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        # 全部小写化
        token_list = [x.lower() for x in cut_tokens]
        # 列表里包含 '' 和' '
        token_list = [x.strip() for x in token_list if x.strip() != '']
        # 返回列表
        return token_list
    # 存在为空的情况，词向量要进行判断
    except:
        return '-1000'
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

def sqlang_query_parse(line):
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)

    # 去除括号
    word_list = [word for word in word_list if not re.findall('[\(\)]', word)]

    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip()]

    return word_list


def sqlang_context_parse(line):
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)

    # 列表里包含 '' 或 ' '
    word_list = [x.strip() for x in word_list if x.strip()]

    return word_list

#######################主函数：句子的tokens##################################


if __name__ == '__main__':
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))



