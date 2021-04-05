
import pandas as pd
import numpy as np

class CommandsPatternsDataset:
    def __init__(self, patterns, tokens_to_keywords):
        """
        :param: patterns: a list of strings containing sentence patterns such as
                          "give me a <ADJECTIVE> <INSTRUMENT>"
        :param: tokens_to_keywords: a dictionart containing tokens as keys and lists 
                                    of possibile keywords for each token as values.
                                    Example:
                                    {
                                        "<ADJECTIVE>" : ["dark","bright","soft","hard","lovely"],
                                        "<INSTRUMENT>" : ["guitar","violin","piano","cello"]
                                    }
        """
        self.patterns = np.array(patterns)
        self.tokens = np.array(sorted(tokens_to_keywords.keys()))
        self.tokens_to_keywords = {k:np.array(v) for k,v in tokens_to_keywords.items()}

    def get_as_df(self):
        """
        This method returns all the possible combinations of patterns and keywords
        by replacing tokens such as "<ADJECTIVE>" with keywords such as "soft"
        in every pattern, with every keyword for each token.
        Additional metadata is provided, such as the original pattern and the position of the 
        keyword that replaced every token.

        returns a dataframe with the following columns
        - sentence: a generated sentence having its tokens replaced by keywords
        - sentence_id: a numeric identifier, unique for each sentence
        - pattern: the pattern that generated the sentence
        - pattern_id: a numeric identifier, unique for each pattern
        - token: one of the tokens that was replaced with a keyword
        - keyword: the keyword that replaced the token
        - start: position of the first character of the keyword that replaced the token
        - end: position of the last character of the keyword that replaced the token
        - token_occurrence: the occurrence of the token that has been replaced by the keyword.
                            when there are token repetitions, such as 
                            "give me a <ADJECTIVE> <ADJECTIVE> <INSTRUMENT>"
                            when token is "<ADJECTIVE>", token_occurrence could be 1 or 2 depending
                            on which occurrence of the <ADJECTIVE> token has been replaced.
        The returned dataframe has multiple rows for each generated sentence. Each row provides 
        informations about one token-to-keyword substitution.
        Example for a single generated sentence:
        pattern                             sentence	            start	end	    token	        keyword
 	    give me a <ADJECTIVE> <INSTRUMENT>  give me a dark guitar	10.0	14.0	<ADJECTIVE>		dark
 	    give me a <ADJECTIVE> <INSTRUMENT>  give me a dark guitar	15.0	21.0	<INSTRUMENT>	guitar
        """
        patterns = self.patterns
        tokens = self.tokens
        tokens_to_keywords = self.tokens_to_keywords
        # generate sentences and associated metadata
        occurrences=np.char.count(patterns,np.expand_dims(tokens,axis=1))
        max_occurrences=occurrences.max(axis=1)
        meta=[]
        res=patterns.copy()
        dimensions=1
        for token, max_occ in zip(tokens, max_occurrences):
            for occurrence in range(max_occ):
                start = np.char.find(res,token).astype(float)
                start[start==-1]=np.nan
                end = start + np.reshape(np.char.str_len(tokens_to_keywords[token]).astype(float),tokens_to_keywords[token].shape + (1,)*dimensions)
                keywords = np.reshape(tokens_to_keywords[token],tokens_to_keywords[token].shape + (1,)*dimensions)
                res=np.char.replace(res,token,keywords,1)
                meta.append({
                    "dimension":dimensions,
                    "start":start,
                    "end":end,
                    "token": token,
                    "token_occurrence": occurrence + 1,
                    "keywords": keywords
                })
                dimensions+=1
        # reshape metadata to fit a dataframe
        new_meta = []
        for m in meta:
            keywords = m["keywords"].reshape((len(res.shape)-len(m["keywords"].shape))*(1,) + m["keywords"].shape)
            rshape = list(res.shape)
            rshape[-len(m["keywords"].shape)] = 1
            keywords = np.char.add(keywords, np.full(rshape, ""))
            start = m["start"].reshape((len(res.shape)-len(m["start"].shape))*(1,) + m["start"].shape)
            rshape = list(res.shape)
            rshape[-len(m["start"].shape)] = 1
            start = start*np.full(rshape, 1)
            end = m["end"].reshape((len(res.shape)-len(m["end"].shape))*(1,) + m["end"].shape)
            rshape = list(res.shape)
            rshape[-len(m["end"].shape)] = 1
            end = end*np.full(rshape, 1)
            new_meta.append({
                "start":start,
                "end":end,
                "token": np.full(res.shape, m["token"]),
                "token_occurrence": np.full(res.shape, m["token_occurrence"]),
                "keyword": keywords
            })
        rshape = list(res.shape)
        rshape[-1] = 1
        patterns_meta = np.char.add(
            patterns.reshape((1,)*(len(res.shape)-1)+patterns.shape),
            np.full(rshape, ""))
        patterns_id_meta = np.arange(patterns.shape[0]).reshape((1,)*(len(res.shape)-1)+patterns.shape) * np.full(rshape, 1)
        # Build a dataframe with row-wise data association
        df_data = [
            ("sentence",np.ravel(res)),
            ("pattern",np.ravel(patterns_meta)),
            ("pattern_id",np.ravel(patterns_id_meta))
        ]
        for count_col,m in enumerate(new_meta):
            for k,v in m.items():
                num = str(count_col).zfill(3)
                df_data.append((f"{k}_{num}",np.ravel(v)))
        df = pd.DataFrame(dict(df_data))
        df = df.iloc[df["sentence"].drop_duplicates().index]
        df = df.reset_index().rename(columns={"index": "sentence_id"})
        # De-normalize dataframe for improved readability and usability 
        cols=["start", "end", "token", "token_occurrence","keyword"]
        denormalized_df =pd.concat([
            df[["sentence_id","sentence","pattern_id","pattern"]+[f"{k}_{str(i).zfill(3)}" for k in cols]].rename(columns=dict([(f"{k}_{str(i).zfill(3)}",k) for k in cols]))
            for i in range(count_col+1)
        ])
        denormalized_df = denormalized_df.dropna()
        return denormalized_df