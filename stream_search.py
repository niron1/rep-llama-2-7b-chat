
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

class DefeatInitialSpaces:
    def __init__(self):
        self.state = {'on_train': False}

    def process(self, s):
        if not self.state['on_train']:
            stripped_s = s.lstrip()
            if stripped_s:
                self.state['on_train'] = True
            return stripped_s
        else:
            return s


def stream_search(needles, streamer):
    un_spacer = DefeatInitialSpaces()
    stream_tag_remover = StreamTagRemover(needles)
    for chunk in streamer:
        for tok in stream_tag_remover.process_chunk(chunk):
            tok1 = un_spacer.process(tok)
            if tok1 != '':
                print("33333",tok1,"33333")
                yield tok1
    for tok in stream_tag_remover.finish():
        tok1 = un_spacer.process(tok)
        if tok1 != '':
            print("44444",tok1,"444444")
            yield tok1


