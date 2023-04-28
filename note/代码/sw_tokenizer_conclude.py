## v2.8.0 BertTokenizer



class BertTokenizer(PreTrainedTokenizer):
    def __init__(self, do_basic_tokenize=True,):
        self.do_basic_tokenize = do_basic_tokenize

    def _tokenize(self, text):
        """
        默认do_basic_tokenize，即先进行按空格的basic_tokenizer；再对每个token进行wordpiece_tokenizer。
        否则直接对输入进行wordpiece_tokenizer
        """
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.all_special_tokens):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens



class WordpieceTokenizer:
    def tokenize(self, text):
        """将token 切分为 word pieces."""



class BasicTokenizer:

    def tokenize(self, text, never_split=None):
        never_split = never_split  # 专有词
        text = self._clean_text(text)  # whitespace字符替换为空格

        text = self._tokenize_chinese_chars(text)   # 在CJK字符前后加空格，其他英文字符等不动
        orig_tokens = whitespace_tokenize(text)     # 先使用空格分隔句子为token-list
        
        split_tokens = []
        for token in orig_tokens:
            split_tokens.extend(self._run_split_on_punc(token, never_split)) # 再对每个token根据标点切分
        output_tokens = whitespace_tokenize(" ".join(split_tokens))  # 最后再将按空格拼接的字符串按空格切分
        return output_tokens


    def _run_split_on_punc(self, text, never_split=None):
        """根据标点切分句子"""

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        - 遍历每一个字符
          - 删除控制字符 (_is_control()判断)
          - 根据_is_whitespace()判断，将空白符替换为 " "
        - return "".join(output:list)
        """


    def _tokenize_chinese_chars(self, text):
        """
        Adds whitespace around any CJK character.
        在每个CJK（中日韩）字符前后各添加一个空格(根据_is_chinese_char()判断)

        return "".join([' ', '中', ' ', ' ', '国', ' ', ' ', '人', ' ', ' ', '民', ' ','a','b','c'])
        """

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character"""



def whitespace_tokenize(text):
    """使用空格分隔句子
    return: list
    """


def _is_whitespace(char):
    """
    判断字符是否为空白
    特例：\t, \n, and \r are technically contorl characters but we treat them as whitespace
    """

def _is_control(char):
    """
    判断字符是否为控制字符
    特例：\t", "\n", "\r" 虽然严格意义上是控制字符，这里我们视为whitespace

    """