"""
Demo of using Memory Network for question answering
"""
import glob
import os
import gzip
import sys
import pickle

import argparse
import numpy as np

from config import BabiConfigJoint
from train_test import train, train_linear_start
from util import parse_babi_task, build_model


class MemN2N(object):
    """
    MemN2N class
    """
    def __init__(self, data_dir, model_file):
        self.data_dir       = data_dir
        self.model_file     = model_file
        self.reversed_dict  = None
        self.memory         = None
        self.model          = None
        self.loss           = None
        self.general_config = None

    def save_model(self):
        with gzip.open(self.model_file, "wb") as f:
            print("Saving model to file %s ..." % self.model_file)
            pickle.dump((self.reversed_dict, self.memory, self.model, self.loss, self.general_config), f)

    def load_model(self):
        # Check if model was loaded
        if self.reversed_dict is None or self.memory is None or \
                self.model is None or self.loss is None or self.general_config is None:
            print("Loading model from file %s ..." % self.model_file)
            with gzip.open(self.model_file, "rb") as f:
                self.reversed_dict, self.memory, self.model, self.loss, self.general_config = pickle.load(f)

    def train(self):
        """
        Train MemN2N model using training data for tasks.
        """
        np.random.seed(42)  # for reproducing
        assert self.data_dir is not None, "data_dir is not specified."
        print("Reading data from %s ..." % self.data_dir)

        # Parse training data
        train_data_path = glob.glob('%s/qa*_*_train.txt' % self.data_dir)
        dictionary = {"nil": 0}
        train_story, train_questions, train_qstory = parse_babi_task(train_data_path, dictionary, False)

        # Parse test data just to expand the dictionary so that it covers all words in the test data too
        test_data_path = glob.glob('%s/qa*_*_test.txt' % self.data_dir)
        parse_babi_task(test_data_path, dictionary, False)

        # Get reversed dictionary mapping index to word
        self.reversed_dict = dict((ix, w) for w, ix in dictionary.items())

        # Construct model
        self.general_config = BabiConfigJoint(train_story, train_questions, dictionary)
        self.memory, self.model, self.loss = build_model(self.general_config)

        # Train model
        if self.general_config.linear_start:
            train_linear_start(train_story, train_questions, train_qstory,
                               self.memory, self.model, self.loss, self.general_config)
        else:
            train(train_story, train_questions, train_qstory,
                  self.memory, self.model, self.loss, self.general_config)

        # Save model
        self.save_model()

    def get_story_texts(self, test_story, test_questions, test_qstory,
                        question_idx, story_idx, last_sentence_idx):
        """
        Get text of question, its corresponding fact statements.
        """
        train_config = self.general_config.train_config
        enable_time = self.general_config.enable_time
        max_words = train_config["max_words"] \
            if not enable_time else train_config["max_words"] - 1

        story = [[self.reversed_dict[test_story[word_pos, sent_idx, story_idx]]
                  for word_pos in range(max_words)]
                 for sent_idx in range(last_sentence_idx + 1)]

        question = [self.reversed_dict[test_qstory[word_pos, question_idx]]
                    for word_pos in range(max_words)]

        story_txt = [" ".join([w for w in sent if w != "nil"]) for sent in story]
        question_txt = " ".join([w for w in question if w != "nil"])
        correct_answer = self.reversed_dict[test_questions[2, question_idx]]

        return story_txt, question_txt, correct_answer

    def predict_answer(self, test_story, test_questions, test_qstory,
                       question_idx, story_idx, last_sentence_idx,
                       user_question=''):
        # Get configuration
        nhops        = self.general_config.nhops
        train_config = self.general_config.train_config
        batch_size   = self.general_config.batch_size
        dictionary   = self.general_config.dictionary
        enable_time  = self.general_config.enable_time

        max_words = train_config["max_words"] \
            if not enable_time else train_config["max_words"] - 1

        input_data = np.zeros((max_words, batch_size), np.float32)
        input_data[:] = dictionary["nil"]
        self.memory[0].data[:] = dictionary["nil"]

        # Check if user provides questions and it's different from suggested question
        _, suggested_question, _ = self.get_story_texts(test_story, test_questions, test_qstory,
                                                        question_idx, story_idx, last_sentence_idx)
        user_question_provided = user_question != '' and user_question != suggested_question
        encoded_user_question = None
        if user_question_provided:
            # print("User question = '%s'" % user_question)
            user_question = user_question.strip()
            if user_question[-1] == '?':
                user_question = user_question[:-1]
            qwords = user_question.rstrip().lower().split() # skip '?'

            # Encoding
            encoded_user_question = np.zeros(max_words)
            encoded_user_question[:] = dictionary["nil"]
            for ix, w in enumerate(qwords):
                if w in dictionary:
                    encoded_user_question[ix] = dictionary[w]
                else:
                    print("WARNING - The word '%s' is not in dictionary." % w)

        # Input data and data for the 1st memory cell
        # Here we duplicate input_data to fill the whole batch
        for b in range(batch_size):
            d = test_story[:, :(1 + last_sentence_idx), story_idx]

            offset = max(0, d.shape[1] - train_config["sz"])
            d = d[:, offset:]

            self.memory[0].data[:d.shape[0], :d.shape[1], b] = d

            if enable_time:
                self.memory[0].data[-1, :d.shape[1], b] = \
                    np.arange(d.shape[1])[::-1] + len(dictionary) # time words

            if user_question_provided:
                input_data[:test_qstory.shape[0], b] = encoded_user_question
            else:
                input_data[:test_qstory.shape[0], b] = test_qstory[:, question_idx]

        # Data for the rest memory cells
        for i in range(1, nhops):
            self.memory[i].data = self.memory[0].data

        # Run model to predict answer
        out = self.model.fprop(input_data)
        memory_probs = np.array([self.memory[i].probs[:(last_sentence_idx + 1), 0] for i in range(nhops)])

        # Get answer for the 1st question since all are the same
        pred_answer_idx  = out[:, 0].argmax()
        pred_prob = out[pred_answer_idx, 0]

        return pred_answer_idx, pred_prob, memory_probs


def train_model(data_dir, model_file):
    memn2n = MemN2N(data_dir, model_file)
    memn2n.train()


def run_console_demo(data_dir, model_file):
    """
    Console-based demo
    """
    memn2n = MemN2N(data_dir, model_file)

    # Try to load model
    memn2n.load_model()

    # Read test data
    print("Reading test data from %s ..." % memn2n.data_dir)
    test_data_path = glob.glob('%s/qa*_*_test.txt' % memn2n.data_dir)
    test_story, test_questions, test_qstory = \
        parse_babi_task(test_data_path, memn2n.general_config.dictionary, False)

    while True:
        # Pick a random question
        question_idx      = np.random.randint(test_questions.shape[1])
        story_idx         = test_questions[0, question_idx]
        last_sentence_idx = test_questions[1, question_idx]

        # Get story and question
        story_txt, question_txt, correct_answer = memn2n.get_story_texts(test_story, test_questions, test_qstory,
                                                                         question_idx, story_idx, last_sentence_idx)
        print("* Story:")
        print("\n\t".join(story_txt))
        print("\n* Suggested question:\n\t%s?" % question_txt)

        while True:
            user_question = raw_input("Your question (press Enter to use the suggested question):\n\t")

            pred_answer_idx, pred_prob, memory_probs = \
                memn2n.predict_answer(test_story, test_questions, test_qstory,
                                      question_idx, story_idx, last_sentence_idx,
                                      user_question)

            pred_answer = memn2n.reversed_dict[pred_answer_idx]

            print("* Answer: '%s', confidence score = %.2f%%" % (pred_answer, 100. * pred_prob))
            if user_question == '':
                if pred_answer == correct_answer:
                    print("  Correct!")
                else:
                    print("  Wrong. The correct answer is '%s'" % correct_answer)

            print("\n* Explanation:")
            print("\t".join(["Memory %d" % (i + 1) for i in range(len(memory_probs))]) + "\tText")
            for sent_idx, sent_txt in enumerate(story_txt):
                prob_output = "\t".join(["%.3f" % mem_prob for mem_prob in memory_probs[:, sent_idx]])
                print("%s\t%s" % (prob_output, sent_txt))

            asking_another_question = raw_input("\nDo you want to ask another question? [y/N] ")
            if asking_another_question == '' or asking_another_question.lower() == 'n': break

        will_continue = raw_input("Do you want to continue? [Y/n] ")
        if will_continue != '' and will_continue.lower() != 'y': break
        print("=" * 70)


def run_web_demo(data_dir, model_file):
    from demo.web import webapp
    webapp.init(data_dir, model_file)
    webapp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", default="data/tasks_1-20_v1-2/en",
                        help="path to dataset directory (default: %(default)s)")
    parser.add_argument("-m", "--model-file", default="trained_model/memn2n_model.pklz",
                        help="model file (default: %(default)s)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-train", "--train", action="store_true",
                       help="train model (default: %(default)s)")
    group.add_argument("-console", "--console-demo", action="store_true",
                       help="run console-based demo (default: %(default)s)")
    group.add_argument("-web", "--web-demo", action="store_true", default=True,
                       help="run web-based demo (default: %(default)s)")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print("The data directory '%s' does not exist. Please download it first." % args.data_dir)
        sys.exit(1)

    if args.train:
        train_model(args.data_dir, args.model_file)
    elif args.console_demo:
        run_console_demo(args.data_dir, args.model_file)
    else:
        run_web_demo(args.data_dir, args.model_file)
