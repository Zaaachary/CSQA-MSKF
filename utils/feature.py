
class Feature:
    def __init__(self, idx, input_ids, input_mask, segment_ids):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

    @classmethod
    def make_single(cls, idx, tokens, tokenizer, max_seq_len):
        '''
        将输入的 token 转化为 ids
        '''
        tokens = ['[CLS]'] + tokens
        tokens = tokens[:max_seq_len-1]
        tokens = tokens + ['[SEP]']

        input_mask = [1] * len(tokens)
        segment_ids = [1] * len(tokens)
        # segment_ids = [0] * question_length

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_len - len(input_ids))
        # segment_padding = [1] * (max_seq_length - question_length)

        input_ids += padding
        input_mask += padding
        segment_ids += padding
        # # segment_ids += segment_padding

        # print("max_seq_length is {}, segment_ids length is {}".format(max_seq_len, len(segment_ids)))

        return cls(idx, input_ids, input_mask, segment_ids)

    def __str__(self):
        return str(self.input_ids)