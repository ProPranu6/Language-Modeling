from collections import defaultdict as dd
from copy import deepcopy
import re

class TextProcessing:                       
      LISTOFCORRECTWORDS = re.split("<br>", open('C:/Users/T.PRANEETH/correct_words_html.txt', 'r', encoding="utf8").read()) 
      


      def __init__(self, allow_stop_words=True, auto_spell_correct=True):
        self.text = None
        self.text_sens = None
        self.sens_tokens = []
        self.tokens_dict = dd(lambda : 0)
        self.ind = 0
        self.vocab_edits = None
        self.auto_spell_correct = auto_spell_correct
        self.allow_stop_words = allow_stop_words
        pass

      @staticmethod
      def add_token(self, tok):
        if self.tokens_dict[tok] == 0:
          self.tokens_dict[tok] = self.ind
          self.ind += 1
        else:
          pass

      @staticmethod
      def add_sens_token(self, sen_tok):
        self.sens_tokens.append(sen_tok)


      def sentence_segmentation(self, text):
        text = text.lower()
        sentence_pattern = "((?<=[^\s.]{3})\s{0,}([.!?])\s{0,}(?=\w*))|\w*:"  #B.R, Dr. Br 
        text_sens = [s.strip() for s in re.sub(sentence_pattern, "<EOS>", text).split("<EOS>") if s.strip() !='']
        return text_sens

      def clean_words(self, word):
        punc = '[.,;:\-\(\)\]\[\$%&\*_!<>@#\\/"=]'
        return re.sub(punc,'', word)

      def tokenization(self, text_sens):
        for sen in text_sens:
          sen_tok = []
          for word in sen.split():
            cleansed_word = self.correct_words(self.clean_words(word)) if self.auto_spell_correct else self.clean_words(word)
            if self.allow_stop_words or not(cleansed_word in (stopwords.words('english') + [''])):
              TextProcessing.add_token(self, cleansed_word)
              sen_tok += [cleansed_word]
          TextProcessing.add_sens_token(self, sen_tok)

      def make_edits(self, vocab, within_edits=2):
        def one_edits(word):
          edits = set()
          for i in range(len(word)):
            stage = list(word)
            stage.pop(i)
            edits.add("".join(stage))
          return edits

        token_edits = {k:[] for k in vocab}
        for word in vocab:
          source = set([word])
          for epoch in range(within_edits):
            new_source = set()
            for wdrep in source:
              temp_edits = one_edits(wdrep)
              new_source = new_source.union(temp_edits)
            source = new_source
            token_edits[word].append(list(new_source))
        return token_edits

      def correct_words(self, raw_word, topn=1):

        top_suggestion = raw_word
        top_dis = 1e+7
        if not(raw_word in self.vocab_edits):
            raw_word_edits = self.make_edits([raw_word], within_edits=1)[raw_word][0]

            used_vocab = {vocab:0 for vocab in self.vocab_edits}
            suggestion_count = 0
            for rwe in raw_word_edits:
              for vocab, ves in self.vocab_edits.items():
                for dis, ve in enumerate(ves):
                  if rwe in ve and not(used_vocab[vocab]) and top_dis>= dis:
                    top_suggestion = vocab
                    top_dis = dis
                    suggestion_count += 1
                    used_vocab[vocab] = 1
                    break

        return top_suggestion



      def __call__(self, text):
        self.text = text

        text_sens = self.sentence_segmentation(text)   #sentence segmentation
        self.text_sens = text_sens 

        if self.vocab_edits == None and self.auto_spell_correct:
          self.vocab_edits = self.make_edits(TextProcessing.LISTOFCORRECTWORDS, within_edits=5)  #creates vocab for spell check with all edits in given range

        self.tokenization(text_sens)  #Tokenization, Punctuation Removal, Stopword Removal and Spell Check



from collections import Counter as c
from collections import defaultdict as d
import numpy as np


class NGrams:


      def __init__(self, train_text, preprocessing_class, N=2, **preprocessing_args):
        self.train_text = train_text
        self.preprocessor_train = preprocessing_class(**preprocessing_args)
        self.preprocessor_test = preprocessing_class(**preprocessing_args)
        self.N = N
        self.preprocessor_train(train_text)
        self.cond_probs = self.generate_probs(self.preprocessor_train.sens_tokens, n=N)




      def generate_n_grams(self, sens_tokens, n=2):
        n_grams = []
        for tokens in sens_tokens:
          while len(tokens) != 0:
            if len(tokens) <n :
              break
            temp = " ".join(tokens[:n])
            n_grams.append(temp)
            tokens = tokens[1:]

        return n_grams

      def generate_probs(self, sens_tokens, n=2):

        n_grams = self.generate_n_grams(sens_tokens, n)
        intotal_phrase_probs = c(n_grams) #word_probs if n==2 else c(generate_n_grams(text, n-1))

        cond_probs = d(lambda : {'<EOS>':1.0})
        focus_words = [phrase.split()[-1] for phrase in n_grams]
        context_words = [" ".join(phrase.split()[:-1]) for phrase in n_grams]
        context_phrase_probs = c(context_words)

        for wds in context_words:
          cond_probs[wds] = dict()

        for cwds in context_words:
          for fwds in focus_words:
            cond_probs[cwds][fwds] = intotal_phrase_probs[cwds+" "+fwds]/(context_phrase_probs[cwds])
        return cond_probs


      def predict_next_word(self, test_text, given_n_words=2, if_print=True, for_gen=False):

        self.preprocessor_test(test_text)
        test_sens_text = [self.preprocessor_test.sens_tokens[-1]]  #most recent sentence of test text since the initiation of Ngrams class
        test_splits = self.generate_n_grams(test_sens_text, given_n_words-1)
        predicted_words = []
        for phrases in test_splits:
          top_words = sorted(list(self.cond_probs[phrases].items()), key=lambda x: x[1], reverse=True)[:5]
          predicted_words.append(top_words[0][0])
          if if_print:
            for predicted_word, score in top_words:
              print(f"\t{predicted_word}: with score {score}")
        
        if for_gen:
            return top_words
        return predicted_words


      def auto_generate(self, until_n_words=5):
        print("AI generated text : \n")
        while 1:
          start_at = input("Start Entering..:")
          if start_at == "<exit>":
            yield 1
          words = 0
          start_at = " ".join(start_at.split()[-self.N+1:])
          n = len(start_at.split())+1
          usgcp = False
          try:  
              while words<until_n_words:
                pwords = self.predict_next_word(start_at, given_n_words=n, if_print=False, for_gen=True)
                pwords_probs = np.array([t[1] for t in pwords])
                pwords_dist = pwords_probs/sum(pwords_probs)
                pword = [np.random.choice([t[0] for t in pwords], p=pwords_dist)]
                if pword[0] == "<EOS>":
                  break
                start_at = start_at.split()[1:] + pword
                print(pword[0], end=" ")
                n = len(start_at)+1
                start_at  = " ".join(start_at)

                words += 1
                if words >=1:
                  usgcp = True
                yield 0
          except:
            print(f"Please make sure your Input has {self.N-1} words at least") 
          print("\n")

      
#processor = TextProcessing()
#processor(text_on_topic)
    
  