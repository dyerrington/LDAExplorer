#!/Users/davidyerrington/virtualenvs/data/bin/python

import logging, gensim, bz2, os, string, glob, os, numpy as np, pandas as pd, sqlite3 as db
from sklearn.preprocessing import scale
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from os import path
from operator import itemgetter
import var_dump
 
class LDAExplorer:

    stopwords_file  =   'conf/stopwords.txt'
    stopwords       =   False
    corpus_files    =   False
    corpus_files_meta = 'data/corpus.meta'
    corpus_data     =   'data/corpus_data.mm'
    document_db     =   'data/docs.db'
    dictionary_data =   'data/dictionary.data'
    model_file      =   'data/model.lda'
    logfile         =   'logs/lda_stats.log' 
    logging         =   {
        'filename': 'lda_stats.log',
        'format':   '%(asctime)s : %(levelname)s : %(message)s',
        'level':    logging.INFO
    }

    number_of_topics =  3

    def __init__(self):

        logging.basicConfig(filename=self.logging['filename'], format=self.logging['format'], level=self.logging['level'])
        self.set_stopwords()
        
        try:
            print "Trying to load corpus files"
            self.corpus_files = pd.read_pickle(self.corpus_files_meta)
            print "Done.."
            # print self.corpus_files.iloc[4567]

        except:
            directory = 'data/' # temp location
            albums  =   [f for f in os.listdir(directory) if os.path.isdir(directory + f)]
            files   =   [glob.glob(directory + file + '/*.txt') for file in albums]
            print files
            self.corpus_files   =   [item.split('data/')[1] for files in files for item in files]
            corpus_files_meta   =   pd.DataFrame(self.corpus_files)
            corpus_files_meta.to_pickle(self.corpus_files_meta)

            print "loading files, saving dataframe ====================================== should not be loading!!!!!!!!!!"
            print corpus_files_meta.head()



    def set_stopwords(self):

        try:
            with open(self.stopwords_file) as fp:
                self.stopwords = set(fp.read().encode('utf-8').replace("\xe2", "").strip().split())

        except IOError as e:
            print "set_stopwords() error: {})".format(e)

    def clean_corpus(self, corpus):

        # remove punctuation table
        # table = string.maketrans("","")
        words = []

        if word.strip() not in cachedStopWords and word.strip() not in stopwords:
            words.append(word)

        return words


    def load_text(self, corpus_file):
        """Loads text file from reference, and returns list of words

        Args:
            text_file: File reference to text file.

        Returns:
            A list of words with punctuation and stopwords removed. 

            For example:

            ['spots', 'roof', 'can', 'dog', 'food']

        Raises:
            IOError: An error occurred accessing the text_file resource.
        """

        try:
            with open(corpus_file) as fp:

                words = []

                # remove punctuation table
                table = string.maketrans("","")

                for line in fp:

                    for word in line.lower().translate(table, string.punctuation).split():

                        if word not in self.stopwords:
                            words.append(word)

                # remove words that occur only once
                tokens_once =   set(word for word in set(words) if words.count(word) == 1)
                words       =   [word for word in words if word not in tokens_once]

            return words

        except IOError as e:
            raise IOError("({})".format(e)) 

    def load_corpus_directory(self, directory):

        # do we have a serialized corpus already?  Try to load that first
        try:
            self.corpus     =   corpora.MmCorpus(self.corpus_data)
            self.dictionary =   corpora.Dictionary.load(self.dictionary_data)
            print "{corpus_data} loaded successfully".format(corpus_data = self.corpus_data)

        except:

            albums  =   [f for f in os.listdir(directory) if os.path.isdir(directory + f)]
            files   =   [glob.glob(directory + file + '/*.txt') for file in albums]

            # flatten that shizz
            self.corpus_files   =   [item for files in files for item in files]
            self.documents      =   [self.load_text(file) for file in self.corpus_files]

            print "%d documents loaded" % len(self.documents)

            self.dictionary     =   corpora.Dictionary(self.documents)
            self.dictionary.save(self.dictionary_data)

            self.corpus         =   [self.dictionary.doc2bow(doc) for doc in self.documents]

            print albums, directory
            # corpus_vectors = (dictionary.doc2bow(doc) for doc in dictionary)
            self.corpus_vectors =   self.dictionary.doc2bow(self.documents[0])

            # save for later..
            gensim.corpora.MmCorpus.serialize(self.corpus_data, self.corpus)

    # TBD:  move lda to self.lda
    def generate_topics_lda(self):
        print "Generating new topics..." 
        lda     =   gensim.models.ldamodel
        
        try:
            model = lda.LdaModel.load(self.model_file)
        except:
            print "Generating new model..."
            
            model = lda.LdaModel(corpus=self.corpus, alpha='auto', id2word=self.dictionary)
            # model = lda.LdaModel(corpus=self.corpus, alpha=.01, id2word=self.dictionary, num_topics=self.number_of_topics, update_every=1, chunksize=1000, passes=4)
            model.save(self.model_file)
            
        scores = []

        for idx, topic in enumerate(model.show_topics(num_topics=self.number_of_topics, num_words=25, formatted=False)): # topn=20
            
            print "Topic #%d:\n------------------------" % idx

            for p, id in topic:
                print p, id.encode('utf-8').strip()
                scores.append([idx, p, id.encode('utf-8').strip()])

            print ""

        scores = pd.DataFrame(scores, columns=['topic', 'score', 'word'])
        print scores
        # scores = np.array(scores)
        # print scores / float(max(scores)) * 100

    def get_topics(self, num_words=5, labels_only=True, topic_id=False):

        print "Getting topics...."

        lda     =   gensim.models.ldamodel
        model   =   lda.LdaModel.load(self.model_file)
        scores  =   {}

        if topic_id:

            topics      =   model.show_topics(num_words=num_words, formatted=False)[int(topic_id)]
            max_prob    =   max(topics, key=itemgetter(0))[0]

            return ((int(round((p / max_prob) * 100)), topic) for p, topic in topics)

        for idx, topic in enumerate(model.show_topics(num_words=num_words, formatted=False)): # topn=20

            if labels_only:
                scores.update({
                    idx: ', '.join([item[1].encode('utf-8').strip() for item in topic])
                })

            else:

                scores.update({
                    idx: topic
                })

        return scores
    
    def get_lda_documents(self, topic_id=5, max_results=35):
        conn    =   db.connect(self.document_db) 
        sql     =   'SELECT topic_id, document_name, ROUND(score * 100, 0) as score FROM topics WHERE topic_id = %d ORDER BY score DESC LIMIT %d' % (int(topic_id), max_results)
        topics  =   pd.read_sql(sql, con=conn)

        # print topics.to_dict('records')
        conn.close()
        return topics.to_dict('records')

    def get_lda_documents_legacy(self, topic_id=0):
        print "Getting documents.................."
        lda     =   gensim.models.ldamodel
        try:
            self.corpus =   corpora.MmCorpus(self.corpus_data)
            model       =   lda.LdaModel.load(self.model_file)
            corpus_lda  =   model[self.corpus]
            # sorted(key=lambda item: -item[1])
            simularity  =   list(enumerate(corpus_lda))

            topics      =   {}

            # print self.corpus_files[:20]

            # Amazing optimization - python rulez #1
            scores = sorted(corpus_lda, reverse=True, key=lambda doc: abs(dict(doc).get(topic_id, 0.0))) 


            # intialize dictionary of topics
            for n in list(xrange(self.number_of_topics)):
                topics.update({n: []})

            # append list of documents in topic
            for (document_id, scores) in simularity[:50]:
                for topic_id, score in scores:
                    # self.corpus_files.iloc[document_id][0]
                    topics[topic_id].append((document_id, (int(round(score * 100)))))

            # sort by score ascending
            for (topic_id, scores) in topics.items():
                topics[topic_id] = sorted(topics[topic_id], key=lambda item: item[1], reverse=True)
            # print model.print_topics(model.num_topics)[topic_id]
            
            return topics[topic_id]

            # print "length of corpus_lda: %d, length of documents array: " % (len(corpus_lda))
            # print self.

        except (RuntimeError, TypeError, NameError):
            print "get_lda_documents() exception not implemented.. %s" % RuntimeError

    def import_document_topics(self, topic_id=1, max_documents=10):
        
        print "Importing document topics to sqlite...."

        lda         =   gensim.models.ldamodel
        self.corpus =   corpora.MmCorpus(self.corpus_data)
        model       =   lda.LdaModel.load(self.model_file)
        corpus_lda  =   model[self.corpus]

        # scores = sorted(corpus_lda, reverse=True, key=lambda doc: abs(dict(doc).get(topic_id, 0.0))) 

        topics = []

        for document_id, row in enumerate(corpus_lda):
            for topic_id, score in row:
                topics.append({
                    'document_id':      document_id, 
                    'document_name':    self.corpus_files.iloc[document_id][0],
                    'topic_id':         topic_id, 
                    'score':            score
                })
        
        conn = db.connect('../data/docs.db') 
        topics = pd.DataFrame(topics)
        print topics.head()
        topics.to_sql('topics', con=conn, if_exists='replace')
        conn.close() 

        # print "score lenght", len(scores)

        # for score in scores[:40]:
            # print score

        # print [(self.corpus_files.iloc[score[0]][0], score[0]) for score in enumerate(scores[:max_documents])]

        # print scores[:max_documents]


    def get_simularities(self, topic_id=1):
        lsi = models.LsiModel(self.corpus, id2word=self.dictionary, num_topics=2)
        vec_lsi = lsi[self.dictionary] # convert the query to LSI space
        print(vec_lsi) 

        index = similarities.MatrixSimilarity(lsi[self.corpus])

print "Loaded library LDAExplorer............"

lda = LDAExplorer()

# lda.get_lda_documents(topic_id=5)

lda.load_corpus_directory('data/')
lda.generate_topics_lda()

# lda.get_topics()
# lda.get_lda_documents()
# lda.import_document_topics()

# print lda.stopwords

# print lda.corpus_vectors
# print lda
