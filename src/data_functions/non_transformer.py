from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from torchtext.vocab import pretrained_aliases


def extract_vocabulary_(data : list(), min_df : int = 1) -> dict:
    
    """
    extracts a vocabulary for non pre-trained language models
    keeps only tokens with > min_documen frequency
    """

    cvec = CountVectorizer(tokenizer=lambda x: x["text"].split(" "), min_df=min_df, lowercase=False)
    cvec.fit_transform(data)
    word_to_ix = cvec.vocabulary_
    
    for word in cvec.vocabulary_:
    
        word_to_ix[word] += 4

    word_to_ix["<PAD>"] = 0
    word_to_ix["<UNKN>"] = 1
    word_to_ix["<SOS>"] = 2
    word_to_ix["<EOS>"] = 3
    
    return word_to_ix

class pretrained_embeds:

    """
    Matches pretrained embeddings to vocabulary
    """
    
    def __init__(self, model : str, ix_to_word : dict):
        
        super(pretrained_embeds, self).__init__()

        """
        model -> model alias for pretrained embeddings (e.g.. "fasttext.simple.300d" or "glove.840B.300d")
        ix_to_word -> dict[id] = word
        """

        self.vectors = pretrained_aliases[model](cache='../.vector_cache')

        self.ix_to_word = ix_to_word

        self.embedding_dim = self.vectors.dim
                
    def processed(self)  -> np.array:
        
        pretrained = np.zeros([len(self.ix_to_word), self.embedding_dim])

        found = 0
        
        for i in range(pretrained.shape[0]):
            
            word = self.ix_to_word[i]
            
            if word in self.vectors.stoi.keys():
        
                pretrained[i,:] = self.vectors[word] 

                found += 1

            elif (word == "<PAD>") or (word == "<SOS>") or (word == "<EOS>"):
        
                pretrained[i,:] = np.zeros(self.embedding_dim)
                
                found += 1
            
            else:
                
                pretrained[i,:] = np.random.randn(self.embedding_dim)

        print("Found ", found, " words out of ", len(pretrained)) 

        return pretrained
   