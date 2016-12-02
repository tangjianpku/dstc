"""End-To-End Memory Networks.
The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

#from base import Model

import tensorflow as tf
import numpy as np
from six.moves import range, reduce
from sklearn import cross_validation, metrics

from .base import Model

'''from reader import TextReader

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

'''

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(Model):
    """End-To-End Memory Network."""
    def __init__(self, session, reader, config,
        initializer=tf.random_normal_initializer(stddev=0.1),
        name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """
        self.reader = reader
        self.dataset = config.dataset
        self.checkpoint_dir = config.checkpoint_dir
        self.config = config

        self._vocab_size = reader.vocab_size

        self._batch_size = config.batch_size
        self._sentence_size = config.sentence_size
        self._memory_size = config.memory_size
        self._embedding_size = config.embedding_size
        self._class_size = config.class_size
        self._hops = config.hops
        self._max_grad_norm = config.max_grad_norm
        self._nonlin = config.nonlin
        self._init = initializer
        self._name = name
        self._random_state = config.random_state
        self._epochs = config.epochs
        self._share_memory_size = config.share_memory_size

        self._evaluation_interval=config.evaluation_interval

        self._build_inputs()
        self._build_vars()

        #self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")
        self._opt = tf.train.AdamOptimizer(learning_rate=config.learning_rate, epsilon=config.epsilon)

        # cross entropy
        self.query_embedding = self._inference(self._stories, self._queries) # (batch_size, vocab_size)

        logits = tf.matmul(self.query_embedding, self.W)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(self._answers, tf.float32), name="cross_entropy")
        
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        #cross_entropy_avg = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
        #print tf
        #self.query_embedding_output = self.query_embedding(self._queries)


        # loss op
        loss_op = cross_entropy_sum
        #loss_op = cross_entropy_avg


        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars if g is not None]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        #self.embedding_op = query_embedding

        init_op = tf.initialize_all_variables()
        self._sess = session
        self._sess.run(init_op)


    def _build_inputs(self):

        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")

        self._answers = tf.placeholder(tf.int32, [None, self._class_size], name="categories")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size]) #### zero padding variable here ...

            A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])  ### Query word embedding  
            B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])  ### Memory Word embedding
            C = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ])  ### Memory Word embedding

            self.A = tf.Variable(A, name="A")
            self.B = tf.Variable(B, name="B")
            self.C = tf.Variable(B, name="C")

            #self.TA = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA')

            #self.H = tf.Variable(self._init([self._hops * self._embedding_size, self._embedding_size]), name="H")
            self.W= tf.Variable(self._init([2 * self._embedding_size, self._class_size]), name="W")

            #self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="W")

            #self.W1 = tf.Variable(self._init([2*self._embedding_size, 512]), name="W")
            #self.W2 = tf.Variable(self._init([512, self._embedding_size]), name="W")

            ##### adding a gating unit here ....
            #self.H = tf.Variable(0, name="H", trainable=True, dtype=tf.float32);

            #self.H1 = tf.nn.sigmoid(self.H)
            #### share memory size

            self.share_memory = tf.Variable(self._init([self._share_memory_size, self._embedding_size]), name="Shared_memory")

        self._nil_vars = set([self.A.name, self.B.name, self.C.name])

    ###################################################  Average word embeddings ###################################

    ##### Approach for calculating query embedding here ####
    def query_embedding_avg(self, queries):
        ### Query input embedding
        q_emb = tf.nn.embedding_lookup(self.A, queries)
        ### approach for learning query embedding
        u = tf.reduce_mean(q_emb, 1)    

        return u
    ##### Approach for calculating memory embedding heree ###

    def memory_embedding_avg(self, stories):
        m_emb = tf.nn.embedding_lookup(self.A, stories)   
        m = tf.reduce_mean(m_emb, 2)

        return m

    ###################################################  GRU ###################################

    def query_embedding_rnn(self, text):
        with tf.variable_scope("rnn_text_encoder"):
            q_emb = tf.nn.embedding_lookup(self.A, text)
            gru_cell =tf.nn.rnn_cell.GRUCell(self._embedding_size)
            state = gru_cell.zero_state(self._batch_size, dtype=tf.float32)
            #state = tf.zeros([self._batch_size, self._embedding_size], dtype=tf.float32)
            #state = state[0:q_emb.get_shape()[0], :]
            for time_step in range(self._sentence_size):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (state, state) = gru_cell(q_emb[:, time_step, :], state)

        return state

    def memory_embedding_rnn(self, text):
        with tf.variable_scope("rnn_text_encoder"):
            m_emb = tf.nn.embedding_lookup(self.A, text)
            m_emb =tf.reshape(m_emb,[-1, self._sentence_size, self._embedding_size])
            
            gru_cell =tf.nn.rnn_cell.GRUCell(self._embedding_size)
            
            state = gru_cell.zero_state(self._batch_size*self._memory_size, tf.float32)

            for time_step in range(self._sentence_size):
                #if time_step > 0:
                tf.get_variable_scope().reuse_variables()
                (state, state) = gru_cell(m_emb[:, time_step, :], state)

            state = tf.reshape(state, [-1, self._memory_size, self._embedding_size])

        return state

    ################################################### CNN ###################################
    
    def query_embedding_cnn(self, text):
        with tf.variable_scope("cnn_text_encoder"):

            q_emb = tf.nn.embedding_lookup(self.A, text)
            q_emb = tf.expand_dims(q_emb, -1)

            pooled_outputs = []
            filter_sizes=list(map(int, self.config.filter_sizes.split(",")))

            num_filters = self._embedding_size

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-max-pool-%s" % filter_size):
                    filter_shape =[filter_size, self._embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(q_emb, W, strides=[1,1,1,1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")
                    pooled = tf.nn.max_pool(h, ksize=[1, self._sentence_size-filter_size +1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
                    pooled_outputs.append(pooled)

            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(3, pooled_outputs)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.config.dropout_keep_prob)

            ### a transformation here ...
            #V = tf.Variable(tf.truncated_normal([num_filters_total, self._embedding_size], stddev=0.1), name="V")
            
            #state =tf.matmul(h_drop, V)
            state = h_drop

        return state


    def memory_embedding_cnn(self, text):

        with tf.variable_scope("cnn_text_encoder"):
            
            m_emb = tf.nn.embedding_lookup(self.A, text)
            m_emb =tf.reshape(m_emb,[-1, self._sentence_size, self._embedding_size])
            m_emb = tf.expand_dims(m_emb, -1)
            pooled_outputs = []
            filter_sizes=list(map(int, self.config.filter_sizes.split(",")))
            num_filters = self._embedding_size

            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-max-pool-%s" % filter_size):
                    filter_shape =[filter_size, self._embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(m_emb, W, strides=[1,1,1,1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")
                    pooled = tf.nn.max_pool(h, ksize=[1, self._sentence_size-filter_size +1, 1, 1], strides=[1, 1, 1, 1], padding="VALID", name="pool")
                    pooled_outputs.append(pooled)

            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(3, pooled_outputs)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.config.dropout_keep_prob)

            #V = tf.Variable(tf.truncated_normal([num_filters_total, self._embedding_size], stddev=0.1), name="V")
            
            #state =tf.matmul(h_drop, V)
            
            state = tf.reshape(h_drop, [-1, self._memory_size, self._embedding_size])

        return state

    def query_embedding(self, queries):

        if self.config.encoder == "avg":
            return self.query_embedding_avg(queries)
        elif self.config.encoder == "rnn":
            return self.query_embedding_rnn(queries)
        elif self.config.encoder == "cnn":
            return self.query_embedding_cnn(queries)
        else:
            print("Unknown encoders")

    def memory_embedding(self, stories):

        if self.config.encoder == "avg":
            return self.memory_embedding_avg(stories)
        elif self.config.encoder == "rnn":
            return self.memory_embedding_rnn(stories)
        elif self.config.encoder == "cnn":
            return self.memory_embedding_cnn(stories)
        else:
            print("Unknown encoders")            
        
    #### memory network here ####
    def _inference(self, stories, queries):

        with tf.variable_scope(self._name):
            u_0 = self.query_embedding(queries)
            u = [u_0]  ###  batch_size * embedding_size
            ##### memory network for reasoing with queries as inputs #####
            out = u_0
            for idx in range(self._hops):
  ### sum of the word embeddings ...., memory embeddings here ...
                # hack to get around no reduce_dot
                #m =self.memory_embedding(stories)  ### batch_size * memory_size * embedding_size 
                m =self.memory_embedding(stories)

                ###
                #m = tf.concat(0, [m, self.share_memory])
                #self.share_memory2 = tf.expand_dims(self.share_memory, 0)

                #m = tf.concat(1, [m, tf.tile(self.share_memory2, [self._batch_size, 1, 1])])  ### batch_size *(memory_size + shared_memory_size)*embed
                #m = tf.tile(self.share_memory2, [self._batch_size, 1, 1])  #with only shared memory
                #m2 =self.memory_embedding2(stories) 
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])  # batch_size * 1 * embed_size 
                dotted = tf.reduce_sum(m * u_temp, 2)   ### batch_size * (memory +shared_memory)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])  ### batch_size *1*  memory
                c_temp = tf.transpose(m, [0, 2, 1])  ### batch * embedding * memory
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                #o_k = tf.reduce_mean(c_temp, 2)

                ### Genearate a new query embedding 
                #u_k = tf.matmul(u[-1], self.H) + o_k
                #u_k = self.H1 * u[-1] + (1.0 - self.H1 ) * o_k
                gru_cell =tf.nn.rnn_cell.GRUCell(self._embedding_size)

                if idx >0 :
                    tf.get_variable_scope().reuse_variables()

                (u_k, u_k)= gru_cell(o_k, u[-1])

                # nonlinearity
                #if self._nonlin:ls
                #    u_k = tf.nn.sigmoid(u_k)

                u.append(u_k)

                #out = out + tf.matmul(u_k, self.H)


                #### do some transformation here ##########
            out = tf.concat(1, [u[0], u[-1]])

            #out = tf. concat(1, u)

            #out = tf.matmul(out, self.H)

            #out1 = tf.matmul(out,self.W1)
            #out1 = tf.nn.relu(out1)
            #out1 = tf.nn.dropout(out1, 0.8)
            ##out2 = tf.matmul(out1, self.W2)
            #out2 = tf.nn.dropout(out2, 0.8)

            return out

    def train(self):

        #S, Q, A = self.reader.train_data

        #testS, testQ, testA = self.reader.test_data
        #data = S + test

        trainS, trainQ, trainA = self.reader.train_data
        #trainS = trainS[0:10000]
        #trainQ = trainQ[0:10000]
        #trainA = trainA[0:10000]

        t_testS, t_testQ, t_testA = self.reader.test_data

        valS, testS, valQ, testQ, valA, testA = cross_validation.train_test_split(t_testS, t_testQ, t_testA, test_size=.9, random_state=self._random_state)
        
        #testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

        #print(testS[0])

        #print len(testS[0])

        print("Training set shape", trainS.shape)

        # params
        n_train = trainS.shape[0]
        n_test = testS.shape[0]
        n_val = valS.shape[0]

        print("Training Size", n_train)
        print("Validation Size", n_val)
        print("Testing Size", n_test)

        train_labels = np.argmax(trainA, axis=1)
        test_labels = np.argmax(testA, axis=1)
        val_labels = np.argmax(valA, axis=1)

        tf.set_random_seed(self._random_state)
        batch_size = self._batch_size

        batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
        batches = [(start, end) for start, end in batches]

        self.val_acc = 0 

        for t in range(1, self._epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                #print "length S", len(s)
                if len(s)!= self._batch_size:
                    continue
                cost_t = self.batch_fit(s, q, a)
                total_cost += cost_t

            if t % self._evaluation_interval == 0:
                '''
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    if len(s)!= self._batch_size:
                        continue    
                    pred = self.predict(s, q)
                    train_preds += list(pred)
                '''    
                train_preds = self.predict(trainS, trainQ)
                val_preds = self.predict(valS, valQ)
                train_acc = metrics.accuracy_score(np.array(train_preds), train_labels[0:len(train_preds)])
                val_acc = metrics.accuracy_score(np.array(val_preds), val_labels[0:len(val_preds)])

                train_f1 = metrics.f1_score(np.array(train_preds), train_labels[0:len(train_preds)], average='macro')
                val_f1 = metrics.f1_score(np.array(val_preds), val_labels[0:len(val_preds)], average='macro')

                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc, train_f1)
                print('Validation Accuracy:', val_acc, val_f1)
                print('-----------------------')
        
                if val_acc > self.val_acc:
                    self.save(self.checkpoint_dir, t)
                    self.val_acc = val_acc 
                    self.val_f1 = val_f1

        ### load the model directory ....
        print("Final Validating Accuracy:", self.val_acc)

        self.load(self.checkpoint_dir)
        test_preds = self.predict(testS, testQ)
        
        test_acc = metrics.accuracy_score(np.array(test_preds), test_labels[0:len(test_preds)])
        test_f1 = metrics.f1_score(np.array(test_preds), test_labels[0:len(test_preds)], average='macro')

        print("Final Testing Accuracy:", test_acc, test_f1)

        #### total queries ###
        #totalQ = np.concatenate((trainQ,t_testQ), axis=0)
        #totalS = np.concatenate((trainS,t_testS), axis=0)
        #self.predict_query_embedding(totalQ, totalS)
        ###

    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _= self._sess.run([self.loss_op, self.train_op], feed_dict=feed_dict)
        return loss

    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        predictions = []
        
        n_train =stories.shape[0]

        for start in range(0, n_train, self._batch_size):
            end = start + self._batch_size
            s = stories[start:end]
            q = queries[start:end]
            if len(s)!= self._batch_size:
                continue    
            pred = self._sess.run(self.predict_op, feed_dict={self._stories: s, self._queries: q})

            predictions += list(pred)

        return predictions

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """

        predictions = []
        
        n_train =stories.shape[0]

        for start in range(0, n_train, self._batch_size):
            end = start + self._batch_size
            s = stories[start:end]
            q = queries[start:end]
            if len(s)!= self._batch_size:
                continue    
            pred = self._sess.run(self.predict_proba_op, feed_dict={self._stories: s, self._queries: q})

            predictions += list(pred)

        return predictions        

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        predictions = []        
        n_train =stories.shape[0]

        for start in range(0, n_train, self._batch_size):
            end = start + self._batch_size
            s = stories[start:end]
            q = queries[start:end]
            if len(s)!= self._batch_size:
                continue    
            pred = self._sess.run(self.predict_log_proba_op, feed_dict={self._stories: s, self._queries: q})

            predictions += list(pred)

        return predictions  

####################################################################################################################
    def predict_query_embedding(self, queries, stories):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        
        predictions = []        
        n_train =stories.shape[0]

        raw_len = 0
        for start in range(0, n_train, self._batch_size):
            end = start + self._batch_size
            s = stories[start:end]
            q = queries[start:end]

            raw_len = len(s)
            
            if len(s)!= self._batch_size:
                s = stories[n_train-self._batch_size: n_train] 
                q = queries[n_train-self._batch_size: n_train] 
            
            pred = self._sess.run(self.query_embedding, feed_dict={self._queries: q, self._stories: s})
            
            if start == 0:
                predictions = pred
                continue
            if raw_len == self._batch_size:
                #predictions += list(pred)
                predictions = np.concatenate((predictions, pred), axis=0)
            else:
                #predictions += list(pred)[self._batch_size-raw_len:self._batch_size]
                predictions = np.concatenate((predictions, pred[self._batch_size-raw_len:self._batch_size, :]), axis=0)

        print predictions.shape

        f = open(self.config.embedding_file, 'w')
        
        f.write(str(predictions.shape[0])+' '+str(predictions.shape[1])+'\n')
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                if j == 0:
                    f.write(str(predictions[i,j]))
                else:
                    f.write(" "+str(predictions[i,j]))
            f.write("\n")

        f.close()
        return predictions   

