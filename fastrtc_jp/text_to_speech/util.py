import re

def remove_code_blocksRE(markdown_text):
    # 正規表現を使用してコードブロックを検出し、それらを改行に置き換えます
    # ```（コードブロックの開始と終了）に囲まれた部分を検出します
    # 正規表現のパターンは、```で始まり、任意の文字（改行を含む）にマッチし、最後に```で終わるものです
    # re.DOTALLは、`.`が改行にもマッチするようにするフラグです
    pattern = r'```.*?```'
    return re.sub(pattern, '\n', markdown_text, flags=re.DOTALL)


def split_to_talk_segments(text:str) -> list[str]:
    """
    音声合成を行う単位に分割する
    基本的に行単位に分割し、先頭だけは句読点単位で分割する
    """
    if text is None or len(text)==0:
        return []
    sz = len(text)
    st = 0
    segments = []
    while st<sz:
        block_start = text.find("```",st)
        newline_pos = text.find('\n',st)
        if block_start>=0 and ( newline_pos<0 or block_start<newline_pos ):
            if st<block_start:
                segments.append( text[st:block_start] )
            block_end = text.find( "```", block_start+3)
            if (block_start+3)<block_end:
                block_end += 3
            else:
                block_end = sz
            segments.append( text[block_start:block_end])
            st = block_end
        else:
            if newline_pos<0:
                newline_pos = sz
            if st<newline_pos:
                segments.append( text[st:newline_pos] )
            st = newline_pos+1
    # 最初の再生の時だけ、先頭の単語を短くする
    firstline:str = segments[0]
    match = re.search(r'[、。！？ 　\'"\(\)\[\]\{\}「」『』（）［］｛｝]', firstline)
    p = match.start() if match else -1
    if 1<p and p<len(firstline)-1:
        segments[0] = firstline[:p+1]
        segments.insert(1, firstline[p+1:])
    return segments