"""
   #%L
   Package name Vex - Vocabulary Extraction System
   %%
   Copyright (C) 2010 - 2018 Department of Veterans Affairs
   %%
   This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
   #L%

Authors: 
  Alec Chapman, 
  Olga Patterson https://github.com/ovpatterson
Relevant citation:
  Velupillai S, Mowery D, Conway M, Hurdle J, Kious B. Vocabulary Development To Support Information Extraction 
   of Substance Abuse from Psychiatry Notes. 2016 [cited 2018 Jan 15];92–101. 
   Available from: http://www.aclweb.org/anthology/W/W16/W16-2912.pdf 
""" 
import re
import string
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


try:
    from pymedtermino.snomedct import SNOMEDCT
except:
    pass

#from IPython.display import clear_output

# Logging info
import logging
from logging.config import fileConfig

#fileConfig('logging_config.ini')
logging.basicConfig(filename='warnings.log', level=logging.DEBUG)
logger = logging.getLogger()
logger.debug('often makes a very good meal of %s', 'visiting tourists')


class BaseLexDiscover():
    """
    This is an abstract class that will contain a few of the base
    functionalities for LexDiscover. Other classes will inherit from this.
    """

    def __init__(self, text='', base_lex=[],
                     sentences=[], vocab=None, pseudolex=[],
                     min_count=1, *args, **kwargs):
        self.raw_text = text
        self.base_lex = [] # This will be the original terms
        self.lex = []
        self.vocab = {} # This will be a dictionary will freq counts
        self.pseudolex = pseudolex
        self.replace_phrases = True
        self.min_count = min_count
        self.set_lex(base_lex)
        #self.stop_words = set(stopwords.words('english'))
        with open('stopwords.txt') as f:
            self.stop_words = set(f.read().splitlines())
        self.stop_words = {''}
        self.num_iterations = 0

        if len(sentences) > 0:
            self.set_sentences_and_vocab(sentences, vocab)
        else:
            self.sentences = self.process_sentences(self.raw_text)
            self.set_vocab()


    def set_sentences_and_vocab(self, sentences, vocab):
        """
        If the text has already been processed, this method will allow the model to initialize
        using the preprocessed text and sentences.
        """
        assert len(self.raw_text) > 0
        assert len(vocab)
        self.sentences = sentences
        self.vocab = vocab
        


    def process_sentences(self, text):
        """
        Takes a single string of text. Lowers the case, replaces any
        instances of multi-word phrases in the corpus with underscored terms,
        tokenizes the text into sentences and words
        and returns a list of sentences.
        """
        print("Processing sentences")
        text = text.lower()
        text = self.sub_phrases(text)
        #stop_words = set(stopwords.words('english'))
        #text = word_tokenize(text)
        #text = [w for w in text if w not in stop_words]
        #text = ' '.join(text)
        sentences = sent_tokenize(text)
        sentences = [word_tokenize(sent) for sent in sentences]
        return sentences

    def sub_phrases(self, text):
        """
        This method substitutes multiple-word phrases with single strings.
        It first replaces any occurences of multiple-word phrases in the
        provided seed lexicon, then automatically creates phrases
        using GenSim's Phraser model.

        Example: If `self.lex` contains the string 'cardiac arrest',
        it will replace all occurences in the corpus with 'cardiac_arrest'.
        It will then automatically find two word phrases.

        TODO: Explore GenSim's Phraser functionality
        TODO: Allow this to iterate more than once for 3+ word phrases
        Also explore how to adjust thresholds
        """

        for i, term in enumerate(set(self.lex).union(self.pseudolex)):
            if not self.is_multi_word(term):
                continue
            unjoined = re.sub('_', ' ', term)
            joined = re.sub(' ', '_', term)
            text = re.sub(unjoined, joined, text)
        return text

        #phrases = Phrases(self.sentences)
        #bigram = Phraser(phrases)

    def sub_phrases_old(self):
        """
        This method substitutes multiple-word phrases with single strings.
        It first replaces any occurences of multiple-word phrases in the
        provided seed lexicon, then automatically creates phrases
        using GenSim's Phraser model.

        Example: If `self.lex` contains the string 'cardiac arrest',
        it will replace all occurences in the corpus with 'cardiac_arrest'.
        It will then automatically find two word phrases.

        TODO: Allow this to iterate more than once for 3+ word phrases
        Also explore how to adjust thresholds
        """

        for i, term in enumerate(set(self.lex).union(self.pseudolex)):
            if len(term.split('_')) > 1:
                #joined = '_'.join(term.split())``
                # Replace in lexicon
                # self.lex[i] = joined
                # Now iterate through the sentences and replace
                for j, sentence in enumerate(self.sentences):
                    self.sentences[j] = self.sub_phrase_in_sentence(term, sentence)
                    #self.sentences[j] = re.sub(term, joined, ' '.join(sentence)).split()

        #phrases = Phrases(self.sentences)
        #bigram = Phraser(phrases)


    def is_multi_word(self, word):
        return len(re.sub('_', ' ', word).split()) > 1


    def sub_phrase_in_sentence(self, joined_phrase, sentence):
        """
        Takes a joined phrase such as 'cardiac_arrest' and replaces
        all instances of 'cardiac arrest' in the sentence
        """
        unjoined = re.sub('_', ' ', joined_phrase)
        sentence = re.sub(unjoined, joined_phrase, ' '.join(sentence)).split()
        return sentence


    def set_lex(self, lex):
        """
        Sets self.lex to be a list of terms from lex. If replace_phrases is True,
        spaces are replaced with underscores.
        """
        if self.replace_phrases:
            self.lex = [re.sub('[\s]+', '_', term) for term in lex]
            self.base_lex = [re.sub('[\s]+', '_', term) for term in lex]
        else:
            self.lex = lex
            self.base_lex = lex


    def set_pseudolex(self, pseudolex):
        if self.replace_phrases:
            self.pseudolex = [re.sub('[\s]+', '_', term) for term in lex]
        else:
            self.pseudolex = pseudolex


    def set_vocab(self):
        for sentence in self.sentences:
            for word in sentence:
                try:
                    self.vocab[word] += 1
                except KeyError:
                    self.vocab[word] = 1


    def add(self, word, override=False):
        """
        Adds a word to the lexicon.
        If the phrase is multi-word and `replace_phrases` is True,
        it adds the two words separately into the lexicon.
        Returns False if a word is already in the lexicon or pseudolex
        or if the word does not occur more than self.min_count.
        If override is True, it adds the term to the lexicon regardless of freq.
        """

        if word in self.lex or word in self.pseudolex:
            return False
        if word in self.stop_words:
            return False

        if (not override) and (self.min_count > 0) and (self.get_count(word) < self.min_count):
            return False

        if len(word.split()) > 1 and self.replace_phrases:
            joined = '_'.join(word.split())
            if joined in self.lex:
                return False
            self.lex.append(joined)
            for sentence in self.sentences:
                self.sub_phrase_in_sentence(word, sentence)
        else:
            self.lex.append(word)
        return True


    def remove(self, term):
        """
        Removes a term from the lexicon.
        Returns True if the term was removed,
        False if it was not.
        """
        try:
            self.lex.pop(term)
            return True
        except ValueError:
            return False


    def get_lex(self):
        """
        Returns the current lexicon as a list. Multi-word phrases are returned
        as multi-word strings, not with an underscore.
        """
        return [term.replace('_', ' ') for term in self.lex]


    def get_base_lex(self):
        return [term.replace('_', ' ') for term in self.base_lex]



    def get_count(self, term):
        """
        Searches the corpus for a term and returns a count.
        Searches the sentences in order to allow for preprocessing.
        Replaces spaces with underscores to allow for multi-word phrases.
        """
        joined = re.sub(' ', '_', term)
        if len(joined.split('_')) > 1 and joined not in self.vocab:
            return self.count_multi_word(term)
        try:
            return self.vocab[joined]
        except KeyError:
            return 0




    def get_counts(self, terms):
        """
        Returns a list of two-tuples of the terms and frequency from terms sorted in
        descending order by frequency.
        """
        counts = [(term, self.get_count(term)) for term in terms]
        return sorted(counts, key=lambda x:x[1], reverse=True)


    def get_lex_counts(self):
        return get_counts(self.get_lex())


    def count_multi_word(self, phrase):
        """
        Searches the sentences for an instance of a multi-word phrase
        that was not in the original vocab. Returns a count.
        """
        count = 0
        phrase = re.sub('_', ' ', phrase)
        for sentence in self.sentences:
            string = ' '.join(sentence)
            search = list(re.finditer(phrase, string))
            count += len(search)
        return count


    def write_lex(self, path, sep='\n'):
        """
        Writes the current lexicon to a file specified by path.
        Sep defines the delimiter, default '\n'
        """
        with open(path, 'w') as f:
            f.write(sep.join(self.get_lex()))
        print("Saved {} terms at {}".format(len(self.lex), path))


    def sort_by_freq(self, to_return, reverse=True):
        return sorted(to_return, key=lambda x:self.get_count(x), reverse=reverse)


    def search_in_sentences(self, term, max_num=-1):
        sents = []
        term_joined = re.sub(' ', '_', term)
        for sent in self.sentences:
            if max_num > 0:
                if len(sents) == max_num:
                    break
            if term_joined in set(sent):
                sents.append(' '.join(sent))
        return sents

    def get_context(self, term, window=(1, 0), remove_stopwords=True):
        """
        Returns a list of N-grams with `term`, their probabilities, and their frequencies.
        A context is defined as the words surrounding the term
        as defined by window, (# terms before, # terms after).
        By default, window is (1, 0), meaning the word before term (bigrams).

        Returns a list of tuples (n-gram, probability, raw count)
        n-gram is a tuple of the full context windows,
        probability is a decimal proportion of frequency,
        raw count is the frequency of the n-gram in the text.
        Sorted in descending order
        """
        term_count = self.get_count(term)
        context_counts = defaultdict(int)
        for i, s in enumerate(self.sentences):
            # If term is multi-word, join it with in an underscore
            s_joined = ' '.join(s)
            term_joined = re.sub(' ', '_', term)
            s_joined = re.sub(term, term_joined, s_joined)
            if term_joined in s_joined.split():
                contexts = self.get_context_window(s_joined.split(), term_joined, window, remove_stopwords)
                for context in contexts:
                    context_counts[context] += 1
        # Create 3-tuples of (term, prob, raw count)
        to_return = [(term, raw_freq/term_count, raw_freq) for (term, raw_freq) in context_counts.items()]
        to_return = sorted(to_return, key=lambda x:x[1], reverse=True)
        return to_return


    def get_context_window(self, s, term, window, remove_stopwords):
        # Remove stopwords from the context windows
        if remove_stopwords:
            sentence = [w for w in s if w not in self.stop_words or w == term]
        else:
            sentence = [w for w in s]
        contexts = [] # A list, in case the term is found more than once
        n_left, n_right = window
        for i, word in enumerate(sentence):
            if word != term:
                continue
            context = []
            # Get all the words to left up to n_left
            start = i - n_left
            for j in range(start, i):
                # If we're outside of the sentence, add 'PHI'
                if j < 0:
                    neighbor = 'PHI'
                else:
                    neighbor = sentence[j]
                context.append(neighbor)

            # Now add the term itself
            context.append(re.sub('_', ' ', term))

            # Get all the words up to n_right inclusive
            end = i + 1 + n_right
            for j in range(i+1, end):
                # If we're outside of the sentence, add 'PHI'
                if j > len(sentence) - 1:
                    neighbor = 'OMEGA'
                else:
                    neighbor = sentence[j]
                context.append(neighbor)
            contexts.append(tuple(context))

        return contexts


    def __str__(self):
        return "{}, \n{} terms in lexicon".format(type(self), len(self.lex))




class LexWord2Vec(BaseLexDiscover):
    """
    Discovers a single synonym for each word in lex using word1vec. Unlike LexSimSeeded,
    this is automatic and requires no user interaction.
    Parameters:
        text: a single string of text representing the preprocessed corpus
        base_lex: a list of one or multi-word vocabulary terms that will be used
            to seed the discovery.
        sim_thresh: the minimum similarity value for a word to be added to the
            lexicon. All words from the corpus with a similarity greater than
            or equal to this value with any term in the lexicon will be added.
            Default 0.5.
        min_count: the minimum times a term must occur in the corpus to be added
            to the lexicon. Default 1.
        model: an optional pretrained Word2Vec model to allow the user to train
            and configure the model manually. Default None. If None, a model is
            trained using text and the default model parameters.
    You can optionally pass in a pre-trained word2vec model
    """

    def __init__(self, text='', base_lex=[], sentences=[], pseudolex=[], neg_categories={},
                vocab={}, sim_thresh=0.5,
                min_count=1, model=None,):
        super().__init__(text=text, base_lex=base_lex, pseudolex=pseudolex, sentences=sentences, vocab=vocab)
        self.set_model(model, min_count)
        self.sim_thresh = sim_thresh
        self.min_count = min_count
        self.neg_categories = neg_categories
        self.neg_terms = [] # Words that are not added because they're more similar to a neg category


    def set_model(self, model=None, min_count=5):
        """
        Trains a word2vec model using default parameters.
        """
        if model:
            if not isinstance(model, Word2Vec):
                raise AssertionError('model must be a pre-trained word2vec \
                model or None.')
            self.model = model
        else:
            self.model = Word2Vec(self.sentences, min_count=min_count)

    def get_model(self):
        """
        Returns the trained word2vec model.
        """
        return self.model


    def discover_lex(self):
        #discovered = self.discover_by_indiv_words()
        discovered = self.discover_by_list()
        for term in discovered:
            self.add(term)
        return self.sort_by_freq(discovered)
        to_return = []
        discovered = sorted(discovered, key=lambda x:x[1], reverse=True)
        #for (term, score) in discovered:
        #    if term in set(self.lex).union(set(self.pseudolex)):
        #        continue
        #    if (score >= self.sim_thresh) and (term not in to_return):
        #        to_return.append(term)
        for term in to_return:
            self.add(term)
        self.num_iterations += 1
        return to_return

    def discover_by_indiv_words(self):
        """
        For each word in self.lex, queries the word2vec model to find
        words with a similarity greater than self.sim_thresh.
        Adds words to the lexicon and returns a list containing the new words.
        TODO: Try using different methods such as using a pseudolex to deliberately
        exclude.
        """

        discovered = []
        for term in self.lex:
            # 100 is just a dummy number to get a large number
            try:
                sim_terms = self.model.wv.similar_by_word(term, topn=100)
                discovered.extend(sim_terms)
            except KeyError: # not in vocabulary
                continue
            #for sim, score in sim_terms:
            #    if score >= self.sim_thresh:
            #        discovered.append((sim, score))
        return discovered


    def discover_by_list(self):
        """
        Checks overall similarity between terms in vocab and all words in the lexicon
        as a list, rather than individual terms.
        """
        discovered = []
        lex = [x for x in self.base_lex if x in self.model.wv.vocab]
        pseudolex = [x for x in self.pseudolex if x in self.model.wv.vocab]
        for i, word in enumerate(self.vocab.keys()):
            if self.get_count(word) < self.min_count:
                continue
            #if i % 100 == 0:
            #    print("word2vec {}/{}".format(i, len(self.vocab)))
            if word in pseudolex or word in lex:
                continue
            try:
                to_compare = [w for w in lex] # To avoid key errors for words that are OOV but in base lex
                sim = self.model.wv.n_similarity([word], to_compare)
            except KeyError as e: # Not in vocabulary
                #raise e
                continue
            except ZeroDivisionError as e: # The word vec list was empty
                return []
            if sim >= self.sim_thresh:
                # Compute scores for neg categories
                neg_scores = []
                for name, neg_words in self.neg_categories.items():
                    to_compare = [w for w in neg_words if w in self.vocab]
                    try:
                        neg_scores.append(self.model.wv.n_similarity([word], to_compare))
                    except ZeroDivisionError as e:
                        print("You need to fix neg categories")
                        raise e
                if not any([x > sim for x in neg_scores]): # If any of the negative categories are more similar, skip it
                    discovered.append(word)
                else:
                    self.neg_terms.append(word)
                    #print("Neg word: {}".format(word))

        return discovered
        #return self.model.wv.most_similar(positive=lex,
        #                                negative=pseudolex, topn=100)







class LexWNLing(BaseLexDiscover):
    """
    Automatically generates linguistic and rule-based lexical variants
    by reordering terms, generating inflection, generating abbreviations,
    and generating misspellings.
    """
    def __init__(self, text='', base_lex=[], sentences=[], vocab={}, min_count=1, edit_dist=2):
        super().__init__(text=text, base_lex=base_lex, sentences=sentences, vocab=vocab, min_count=min_count)
        self.edit_dist = edit_dist


    def discover_lex(self):
        n_away = self.edit_n_away(self.lex, n=self.edit_dist)
        abbr = self.gen_abbreviations(self.lex)
        new_lex = list(n_away.union(abbr))
        for term in new_lex:
            self.add(term)
        return new_lex


    def edit_n_away(self, words, n=1):
        """
        This method generates new strings n steps away from the original word.
        Takes words, a list of words to manipulate, and n, the number of steps of editing.
        Uses substitution, deletion, and insertion. Underscores (ie., connectors for
        multi-word phrases) only use deletion.
        Returns a list of words that are above the min_count threshold.
        """

        assert(isinstance(words, list))

        queue = {} # A dictionary that will have a queue for each step in n
        queue[0] = []
        queue[0].extend(words)
        gen_words = []
        num_iter = 0
        for i in range(1, n):
            queue[i] = []

        while(num_iter < n):
            while(len(queue[num_iter]) > 0):
                word = queue[num_iter].pop(0)
                if len(word) <= 3: # Don't edit for short words, too much junk
                    continue
                subbed = self.substitute(word)
                deleted = self.deletion(word)
                inserted = self.insertion(word)
                for gen in subbed.union(deleted).union(inserted):
                    if gen in self.lex:
                        continue
                    try:
                        if self.vocab[gen] >= self.min_count:
                            gen_words.append(gen)
                    except KeyError:
                        pass
                    if num_iter + 1 != n and gen not in queue[num_iter + 1]: # If there's another step, add it to the next queue
                        queue[num_iter + 1].append(gen)
            num_iter += 1
        return set(gen_words)


    def substitute(self, word):
        """
        For each letter in a word, substitutes one other letters a-z.
        If the letter is an underscore used to join a multi-word phrase,
        that letter is passed over.
        Returns a unique set of generated words
        """
        subbed = []
        for i in range(len(word)):
            if word[i] == '_':
                continue
            for letter in string.ascii_lowercase:
                gen_word = word[:i]
                if letter == word[i]:
                    continue
                gen_word += letter
                gen_word += word[i+1:]
                if gen_word in self.lex:
                    continue
                subbed.append(gen_word)
        return set(subbed)


    def deletion(self, word):
        """
        For each letter in a word, returns a copy of the word with the letter deleted.
        Returns a unique set of generated words.
        """
        deleted = []
        for i in range(len(word)):
            gen_word = word[:i] + word[i+1:]
            deleted.append(gen_word)
        return set(deleted)


    def insertion(self, word):
        """
        For each index of word, this inserts a letter before that letter.
        Enters one letter at the end.
        Returns a unique set of generated words.
        """
        inserted = []
        for i in range(len(word)):
            for letter in string.ascii_lowercase:
                gen_word = word[:i]
                gen_word += letter
                gen_word += word[i:]
                inserted.append(gen_word)
        for letter in string.ascii_lowercase:
            gen_word = word
            gen_word += letter
            inserted.append(gen_word)
        return set(inserted)


    def gen_abbreviations(self, words):
        """
        This method returns a list of possible abbreviations and truncations
        by splicing the first 3-4 letters of a single word and removing vowels (after the first letter)
        or by taking the first letters of a multi-word phrase after checking that it is not a stop word.
        Returns a set of possible words that occur at least min_count times.
        """
        assert(isinstance(words, list))
        abbreviations = []
        for word in words:
            if self.is_multi_word(word):
                gen_words = self.gen_abbr_multi_words(word)
            else:
                gen_words = self.gen_abbr_single_word(word)
            for gen in gen_words:
                if gen in self.stop_words:
                    continue
                if len(gen) == 1: # TODO: Why are some single letters being returned?
                    continue
                if gen not in self.vocab:
                    continue
                if self.vocab[gen] >= self.min_count:
                    abbreviations.append(gen)
        return set(abbreviations)


    def gen_abbr_single_word(self, word):
        first_three = word[:3]
        first_four = word[:4]
        no_vowels = word[0] + re.sub('[aeiouy]+', '', word[1:])
        return [first_three, first_four, no_vowels]


    def gen_abbr_multi_words(self, word):
        """
        Generates an abbreviaton using the first letter of each word
        and combinations of abbreviations of each single word.
        """
        abbrevs = []
        unjoined = re.sub('_', ' ', word)
        acronym = ''.join([w[0] for w in unjoined.split()]) # Simple acronym, just the first letter of each word
        abbrevs.append(acronym)

        single_abbrs = [self.gen_abbr_single_word(word) for word in unjoined.split()]
        for group in zip(*single_abbrs):
            abbr = ' '.join(group)
            abbrevs.append(abbr)

        return set(abbrevs)




class LexOnt(BaseLexDiscover):
    """
    Finds synonyms using SNOMED-CT
    :param codes - a list of SNOMED-CT codes as ints
    """
    def __init__(self, text='', sentences=[], base_codes=[], vocab=[], min_count=1, parents=False, children=False):
        super().__init__(text=text, base_lex=[], sentences=sentences, vocab=vocab, min_count=min_count)
        self.codes = base_codes
        self.base_codes = base_codes
        self.codes = [x for x in base_codes]
        self.base_lex = []
        self.set_lex_with_codes(base_codes)

        self.parents = parents
        self.children = children



    def set_lex_with_codes(self, codes):
        """
        Adds the primary term in SNOMED for each code in the base codes
        """
        for code in codes:
            concept = SNOMEDCT[code]
            #self.base_lex.append(concept.term.lower())
            self.base_lex.append(concept.term.lower())
            self.add(concept.term.lower(), override=False) # For this implementation, don't worry about frequency count

    def discover_lex(self):
        """
        Calls `_discover_lex` unless there are any errors due to a build error with pymedtermino,
        in which case it returns an empty list.
        :return:
        """
        try:
            to_return = self._discover_lex()
        except OperationalError as e:
            print(e)
            print("Please check that pymedtermino was built correctly.")
            to_return = []
        return to_return

    def _discover_lex(self):
        """
        Finds all synonyms for a concept.
        :return: list of synonyms that occur in the corpus at least min_count times
        """

        # First add optional parents and children
        initial_codes = [x for x in self.codes]
        if self.parents == True or self.children == True:
            for code in initial_codes:
                concept = SNOMEDCT[code]
                if self.parents:
                    self.codes.extend([p.code for p in concept.parents])
                if self.children:
                    self.codes.extend([c.code for c in concept.children])

        to_return = []
        for code in self.codes:
            concept = SNOMEDCT[code]
            synset = concept.terms
            for syn in synset:
                was_added = self.add(syn.lower())
                if was_added:
                    to_return.append(syn.lower())

        return to_return

    def get_codes(self):
        return self.codes

    def get_base_codes(self):
        return self.base_codes




class AggregateLexDiscover(BaseLexDiscover):
    """
    This class is initiated and does all of the text processing.
    It can then accept other models as arguments and aggregate their results.
    """
    def __init__(self, text='', sentences=[], vocab={}, base_lex=[], pseudolex=[], base_codes=[],
                 min_count=1, edit_dist=2, models={}, sim_thresh=0.5,
                 children=True, parents=True,
                 neg_categories={}, **kwargs):
        self.neg_categories = neg_categories
        if neg_categories!= {}:
            for name, words in neg_categories.items():
                pseudolex.extend(words)
        super().__init__(text=text, sentences=sentences, vocab=vocab, base_lex=base_lex, pseudolex=pseudolex,
                         min_count=min_count, )
        self.edit_dist=edit_dist
        self.models = {}
        self.replace_phrases=True
        self.base_codes = base_codes
        self.sim_thresh = sim_thresh
        self.children = children
        self.parents = parents
        self.kwargs = kwargs
        if models == {} or isinstance(models, list): # If models is either an empty dictionary or a list of strings
            self.from_default(models)
        elif ( isinstance(models, dict)):
            if ('word2vec' in models.keys()):
                self.models['word2vec'] = LexWord2Vec(text=self.raw_text,
                                                      base_lex=self.lex,
                                                      sentences=self.sentences,
                                                      neg_categories=self.neg_categories,
                                                      sim_thresh=self.sim_thresh,
                                                      vocab=self.vocab,
                                                      pseudolex=self.pseudolex,
                                                      min_count=self.min_count,
                                                      model = models['word2vec'])
                mods = ['ont', 'wnling']
                self.from_default(mods)

    def from_default(self, models=[]):
        """
        This method adds default models to self.models.
        """
        if len(models) == 0:
            models = ['word2vec', 'wnling']
        try:
            from pymedtermino.snomedct import SNOMEDCT
            models.append('ont')
        except ImportError as e:
            print("Error reading PyMedTermino package")

        if 'word2vec' in models and self.sim_thresh > 0:
            self.models['word2vec'] = LexWord2Vec(text=self.raw_text, 
                                         base_lex=self.lex, 
                                         sentences=self.sentences,
                                         neg_categories=self.neg_categories, 
                                         sim_thresh=self.sim_thresh, 
                                         vocab=self.vocab,
                                         pseudolex=self.pseudolex,  
                                         min_count=self.min_count)
        if 'wnling' in models:
            self.models['wnling'] = LexWNLing(text=self.raw_text, 
                                         base_lex=self.lex, 
                                         sentences=self.sentences,
                                         vocab=self.vocab, 
                                         min_count=self.min_count, 
                                         edit_dist=self.edit_dist)
        if 'ont' in models:
            self.models['ont'] = LexOnt(text=self.raw_text, 
                                        sentences=self.sentences, 
                                        vocab=self.vocab,
                                        min_count=self.min_count,
                                        base_codes=self.base_codes, 
                                        children=self.children,
                                        parents=self.parents)

    def discover_lex(self, intersection=False):
        """
        Calls discover_lex on each child model. If intersection==True, it will only return
        new terms that were found by at least two of the models.
        """
        new_terms = {}
        for model_name, model in self.models.items():
            if model_name == 'wnling':
                print("Discovering linguistic variants")
            elif model_name == 'ont':
                print("Discovering SNOMED synonyms and related concepts")
            elif model_name == 'word2vec':
                print("Discovering similar words with word2vec")
            model_new_terms = model.discover_lex()
            new_terms[model_name]= model_new_terms
        all_new_terms = []
        to_return = []
        if not intersection or len(self.models) == 1:
            for list_of_words in new_terms.values():
                all_new_terms.extend(list_of_words)
            for i, term in enumerate(all_new_terms):
                if self.add(term):
                    to_return.append(term)
            return self.sort_by_freq(to_return)
            #return sorted(to_return, key=lambda x:self.vocab[x], reverse=True)

        to_return = []
        for model_name, new_words in new_terms.items():
            for other_model_name, other_new_words in new_terms.items():
                if model_name == other_model_name:
                    continue
                to_return.extend(set(new_words).intersection(set(other_new_words)))
        to_return = list(set(to_return))
        added_words = []
        for term in to_return:
            if self.add(term):
                added_words.append(term)
        return self.sort_by_freq(added_words)

    def get_intersect(self, *args):
        to_return = []
        model_names = args
        for model_name in model_names:
            for other_model_name in model_names:
                if model_name == other_model_name:
                    continue
                intersect = set(self.models[model_name].lex
                                ).intersection(self.models[other_model_name].lex)
                to_return.extend(intersect)
        to_return = list(set(to_return).difference(set(self.base_lex)))
        return self.sort_by_freq(to_return)

    def get_codes(self):
        if 'ont' in self.models:
            return self.models['ont'].get_codes()
        else:
            return False

    def get_base_codes(self):
        if 'ont' in self.models:
            return self.models['ont'].get_base_codes()
        else:
            return False


    def get_union(self, *args):
        """
        Returns the union between the model names specified in *args.
        For example, passing in 'word2vec' and 'wnling' will return all of
        the vocabulary terms that were discovered by the word2vec and ling models
        but not in the original lexicon.
        """
        to_return = []
        if len(args):
            model_names = args
        else:
            model_names = self.models.keys()
        for model_name in model_names:
            for other_model_name in model_names:
                if model_name == other_model_name:
                    continue
                union = set(self.models[model_name].get_lex()
                                ).union(self.models[other_model_name].get_lex())
                to_return.extend(union)
        to_return = list(set(to_return).difference(set(self.get_base_lex())))
        return self.sort_by_freq(to_return)


    def get_difference(self, first_model, *args):
        """
        Returns the values that are in first_model's lex and none of the others.
        """
        if not len(args):
            other_models = [x for x in self.models.keys() if x != first_model]
        else:
            other_models = args

        # Candidate terms are the first model's lexicon
        candidates = set(self.models[first_model].get_lex())
        # Exclude any terms that are intersect between candidates and the other models' lex
        to_exclude = []
        for model in other_models:
            intersect = candidates.intersection(set(self.models[model].get_lex()))
            to_exclude.extend(intersect)

        # Now find the difference between candidates and to_exclude
        to_return = list(candidates.difference(set(to_exclude)))
        # Exclude baseline terms
        to_return = list(set(to_return).difference(set(self.get_base_lex())))
        return self.sort_by_freq(to_return)


    def get_discovered_terms(self, model_name=None):
        """
        Returns all new lexical terms
        """
        if model_name:
            to_return = list(set(self.models[model_name].get_lex()).difference(set(self.get_base_lex())))
            return self.sort_by_freq(to_return, reverse=True)
        return self.sort_by_freq(self.get_union())

    #def get_sim(self, term, other=None):
    #    return self.models.get_sim(t, other)

    def get_sim(self, term, other=None):
        """
        Returns the similarity between a term and other.
        Other is either a list or a single word.
        :param term:
        :return:
        """
        if not other:
            other = self.base_lex
        if 'word2vec' not in self.models:
            raise ValueError("There is no word2vec model instantiated.")
        if not isinstance(term, str):
            raise ValueError("term must be a string")

        # Take out any words that were provided that weren't found in the corpus, so aren't in the vocab
        other = [x.replace(' ', '_') for x in other if self.get_count(x) > self.min_count]
        for o in other:
            try:
                assert o in self.models['word2vec'].model.wv
            except:
                return "Failed: {}".format(o)
                raise e


        wv = self.models['word2vec'].model.wv
        if isinstance(other, str):
                try:
                    return wv.similarity(term, other)
                except KeyError as e: # out of vocab
                    logging.warning("Found word that is not in vocabulary %s . Returning sim score of 0", term)
                    return 0
                    raise e
                    return term
        if (isinstance(other, list) or isinstance(other, set)
                or isinstance(other, tuple)):
            try:
                return wv.n_similarity([term], other)
            except KeyError as e:
                logging.warning("Found word that is not in vocabulary %s . Returning sim score of 0", term)
                return 0
                raise e
                return term
                return 0
        else:
            raise ValueError("other must be either a list, set, or tuple")

    def sort_by_sim(self, to_sort=[], n=-1):
        """
        Sorts the given list. If no list is provided, then it sorts discovered terms.
        n is the number of top terms to consider. Default -1 will return all values in to_sort
        """
        if to_sort == []:
            to_sort = self.get_discovered_terms()
        if n == -1:
            n = len(to_sort)
        scored_terms = [(t, self.get_sim(t)) for t in to_sort]
        scored_terms = list(sorted(scored_terms, key=lambda x:x[1], reverse=True))
        return scored_terms


    def get_neg_words(self):
        """
        Returns any words that were retrieved by word2vec but not added because they were similar to a negative category.
        :return:
        """
        return self.models['word2vec'].neg_terms


    def __str__(self):
        string = super(AggregateLexDiscover, self).__str__()
        string += "\n{} models: {}".format(len(self.models), self.models)
        return string




if __name__ == '__main__':

    corpus = 'There is a fluid collection in the abdomen. There is concern for abscess\
        near the liver. There is no hematoma.'
    sentences = [word_tokenize(sent) for sent in sent_tokenize(corpus)]
    lex = ['abscess', 'hematoma', 'fluid collection']
    #model = Word2Vec(sentences, min_count=1)
    #print(model)
    #fc_discover = LexWord2Vec(corpus, lex, model_min_count=1)
    #discovered = fc_discover.discover_lex()
