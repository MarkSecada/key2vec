from .constants import STOPWORDS, POS_BLACKLIST

class Cleaner(object):
    """Cleans candidate keyphrase"""

    def __init__(self, doc):
        self.doc = doc
        self.tokens = [token for token in doc]

    def transform_text(self):
        transformed_text = []
        tokens_len = len(self.tokens)
        for i, token in enumerate(self.tokens):
            remove = False
            if (i == 0) or (i == tokens_len - 1):
                #is_determiner = self.__is_derminer(token)
                is_stop = self.__is_stop(token)
                is_banned_pos = self.__is_banned_pos(token)
                #is_common = self.__is_common(token)
                remove = (is_stop or is_banned_pos)
            else:
                pass
            if not remove:
                transformed_text.append(token.text)

        if transformed_text == []:
            return ''
        else:
            return ' '.join(transformed_text)

    def __is_stop(self, token):
        return token.text in STOPWORDS

    def __is_banned_pos(self, token):
        return token.pos_ in POS_BLACKLIST

    def __is_determiner(self, token):
        pass

    def __is_common(self, token):
        pass