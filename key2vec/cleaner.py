from .constants import STOPWORDS, POS_BLACKLIST, DETERMINERS

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
                is_stop = token.text in STOPWORDS
                is_banned_pos = token.pos_ in POS_BLACKLIST
                is_determiner = token.text in DETERMINERS
                remove = (is_stop or is_banned_pos or is_determiner)
            else:
                pass
            if not remove:
                transformed_text.append(token.text)

        if transformed_text == []:
            return ''
        else:
            return ' '.join(transformed_text).lower()