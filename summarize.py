
from typing import List
from konlpy.tag import Okt
from lexrankr import LexRank

class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens

def sum(body) :
    a = ""
    okttokenizer: OktTokenizer = OktTokenizer()
    lexrank: LexRank = LexRank(okttokenizer)

    lexrank.summarize(body)
    for i in lexrank.probe(.2) :
        a += i
    return a


if __name__ == "__main__" :
    pass