
class StreamTagRemover:
    def __init__(self, tags):
        self.tags = tags
        self.buffer = ""
        self.result = ""

    def process_chunk(self, chunk):
        self.buffer += chunk
        for tag in self.tags:
            while tag in self.buffer:
                index = self.buffer.find(tag)
                valid_piece = self.buffer[:index]
                yield valid_piece
                self.result += valid_piece
                self.buffer = self.buffer[index + len(tag):]
        valid_piece = self.buffer[:len(self.buffer) - max(len(tag) for tag in self.tags)]
        yield valid_piece
        self.result += valid_piece
        self.buffer = self.buffer[len(self.buffer) - max(len(tag) for tag in self.tags):]

    def finish(self):
        valid_piece = self.buffer
        yield valid_piece
        self.result += valid_piece
        return self.result

# chunks = ["<s>","This is","<s",">"," an interesting line ","</","s>","with so many</s> letters","</s>"]


def stream_search(needles, streamer):
    stream_tag_remover = StreamTagRemover(needles)
    for chunk in streamer:
        for tok in stream_tag_remover.process_chunk(chunk):
            if tok != '':
                yield tok
    for tok in stream_tag_remover.finish():
        if tok != '':
            yield tok

# for new_text in stream_search():
#     print("llm says:",f'[{new_text}]')



