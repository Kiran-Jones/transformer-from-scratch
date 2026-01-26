
import regex as re



class Tokenizer:
    """BPE Tokenizer implementation."""

    def __init__(self, encoder=None, merges=None):
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        self.encoder = encoder if encoder else {}
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.merges = merges if merges else {}
        
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        self.cache = {}

        self.special_tokens = ["<pad>", "<SOS>", "<EOS>"]

        for token in self.special_tokens:
            if token not in self.encoder:
                index = len(self.encoder)
                self.encoder[token] = index
                self.decoder[index] = token

    def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pattern, text):
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes) 

            bpe_segments = self.bpe(token_translated).split(' ')   
            bpe_tokens.extend(self.encoder.get(bpe_token, 0) for bpe_token in bpe_segments)
            
        return bpe_tokens
    
    def decode(self, tokens):
        text_buffer = []
        byte_array = bytearray()

        for token in tokens:
            token_str = self.decoder.get(token, '')

            if token_str in self.special_tokens:
                if byte_array:
                    text_buffer.append(byte_array.decode('utf-8', errors='replace'))
                    byte_array = bytearray()
                text_buffer.append(token_str)
            else:

                for c in token_str:
                    if c in self.byte_decoder:
                        byte_array.append(self.byte_decoder[c])
                    else:
                        byte_array.extend(c.encode('utf-8'))

        if byte_array:
            text_buffer.append(byte_array.decode('utf-8', errors='replace'))

        return "".join(text_buffer)
    

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token
        
        while True:
            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            if bigram not in self.merges:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    @staticmethod
    def get_pairs(word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
        cs = [chr(n) for n in bs] 
        
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(chr(256 + n))
                n += 1
        
        return dict(zip(bs, cs))
    
    @classmethod
    def from_files(cls, encoder_path, merges_path):
        import json
        
        with open(encoder_path, 'r', encoding='utf-8') as f:
            encoder = json.load(f)

            
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_list = json.load(f)
            merges = {tuple(pair): i for i, pair in enumerate(merges_list)}
            
        return cls(encoder=encoder, merges=merges)
    