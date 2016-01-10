import numpy as np

class BabiConfig(object):
    """
    Configuration for bAbI
    """
    def __init__(self, train_story, train_questions, dictionary):
        self.dictionary       = dictionary
        self.batch_size       = 32
        self.nhops            = 3
        self.nepochs          = 100
        self.lrate_decay_step = 25   # reduce learning rate by half every 25 epochs

        # Use 10% of training data for validation
        nb_questions       = train_questions.shape[1]
        nb_train_questions = int(nb_questions * 0.9)

        self.train_range    = np.array(range(nb_train_questions))
        self.val_range      = np.array(range(nb_train_questions, nb_questions))
        self.enable_time    = True   # add time embeddings
        self.use_bow        = False  # use Bag-of-Words instead of Position-Encoding
        self.linear_start   = True
        self.share_type     = 1      # 1: adjacent, 2: layer-wise weight tying
        self.randomize_time = 0.1    # amount of noise injected into time index
        self.add_proj       = False  # add linear layer between internal states
        self.add_nonlin     = False  # add non-linearity to internal states

        if self.linear_start:
            self.ls_nepochs          = 20
            self.ls_lrate_decay_step = 21
            self.ls_init_lrate       = 0.01 / 2

        # Training configuration
        self.train_config = {
            "init_lrate"   : 0.01,
            "max_grad_norm": 40,
            "in_dim"       : 20,
            "out_dim"      : 20,
            "sz"           : min(50, train_story.shape[1]),  # number of sentences
            "voc_sz"       : len(self.dictionary),
            "bsz"          : self.batch_size,
            "max_words"    : len(train_story),
            "weight"       : None
        }

        if self.linear_start:
            self.train_config["init_lrate"] = 0.01 / 2

        if self.enable_time:
            self.train_config.update({
                "voc_sz"   : self.train_config["voc_sz"] + self.train_config["sz"],
                "max_words": self.train_config["max_words"] + 1  # Add 1 for time words
           })


class BabiConfigJoint(object):
    """
    Joint configuration for bAbI
    """
    def __init__(self, train_story, train_questions, dictionary):

        # TODO: Inherit from BabiConfig
        self.dictionary       = dictionary
        self.batch_size       = 32
        self.nhops            = 3
        self.nepochs          = 60

        self.lrate_decay_step = 15   # reduce learning rate by half every 25 epochs  # XXX:

        # Use 10% of training data for validation  # XXX
        nb_questions        = train_questions.shape[1]
        nb_train_questions  = int(nb_questions * 0.9)

        # Randomly split to training and validation sets
        rp = np.random.permutation(nb_questions)
        self.train_range = rp[:nb_train_questions]
        self.val_range   = rp[nb_train_questions:]

        self.enable_time    = True   # add time embeddings
        self.use_bow        = False  # use Bag-of-Words instead of Position-Encoding
        self.linear_start   = True
        self.share_type     = 1      # 1: adjacent, 2: layer-wise weight tying
        self.randomize_time = 0.1    # amount of noise injected into time index
        self.add_proj       = False  # add linear layer between internal states
        self.add_nonlin     = False  # add non-linearity to internal states

        if self.linear_start:
            self.ls_nepochs          = 30  # XXX:
            self.ls_lrate_decay_step = 31  # XXX:
            self.ls_init_lrate       = 0.01 / 2

        # Training configuration
        self.train_config = {
            "init_lrate"   : 0.01,
            "max_grad_norm": 40,
            "in_dim"       : 50,  # XXX:
            "out_dim"      : 50,  # XXX:
            "sz"           : min(50, train_story.shape[1]),
            "voc_sz"       : len(self.dictionary),
            "bsz"          : self.batch_size,
            "max_words"    : len(train_story),
            "weight"       : None
        }

        if self.linear_start:
            self.train_config["init_lrate"] = 0.01 / 2

        if self.enable_time:
            self.train_config.update({
                "voc_sz"   : self.train_config["voc_sz"] + self.train_config["sz"],
                "max_words": self.train_config["max_words"] + 1  # Add 1 for time words
           })
