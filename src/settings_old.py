# 데이터 상대 경로
import datetime
import json
import uuid
import argparse

# Data를 불러올 위치
DATAPATH = "../datasets"
# Config를 불러올 위치
CONFIGPATH = "../config"


class TrainResult:
    """실험하면서 각종 수치를 저장한다."""

    def __init__(self):
        self.total_iter = 0
        self.avg_loss = 5.0

        self.best_test_AUC = 0
        self.best_test_epoch = 0
        self.test_AUC_list = []
        self.test_accuracy_list = []

        self.best_val_AUC = 0
        self.best_val_epoch = 0
        self.val_AUC_list = []
        self.val_accuarcy_list = []
        self.loss_list = []

        self.train_size = 0
        self.val_size = 0
        self.test_size = 0
        self.total_size = 0

    def set_sizes(self, train_size, val_size, test_size):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.total_size = train_size + val_size + test_size

    def to_dict(self):
        return self.__dict__


#parser = argparse.ArgumentParser()
#parser.add_argument("--name", default="exp")
#parser.add_argument("--seed", type=int, default=0)
#parser.add_argument("--batch_size", type=int, default=64)
#parser.add_argument("--init_lr", type=float, default=1e-1)
#parser.add_argument("--lr_decay", type=float, default=0.99)
#parser.add_argument("--momentum", type=float, default=0.9)
#parser.add_argument("--num_folds", type=int, default=10)
#parser.add_argument("--weight_decay", type=float, default=1e-6)
#parser.add_argument("--model_name", default="ClassifierWithBatchNorm")
#parser.add_argument("--optimizer", default="Adadelta")
#parser.add_argument("--num_epoch", type=int, default=50)
#parser.add_argument("--activation", type=str, default="tanh")
#parser.add_argument("--use_data_dropout", action="store_true")

opt = parser.parse_args()

    
class TrainInformation:
    """실험에 사용할 세팅을 저장한다."""

    
    NAME = "name v3_actswish_bs4096_largercapa"
    SEED = opt.seed
    BS = 4096
    INIT_LR = 0.100000
    LR_DECAY = 0.999
    MOMENTUM = opt.momentum
    FOLD = 10
    WEIGHT_DECAY = opt.weight_decay
    MODEL_NAME = "ClassifierWithEmbedding"
    # NCHS = [512, 512, 512, 512, 1]
    # NCHS = [4096, 1]
    #NCHS = [512, 512, 1]
    NCHS = [2048, 2048, 2048, 512, 1]
    ACTIVATION = swish
    # NCHS = [512, 1]
    TRAIN_UID = f"{uuid.uuid4()}"
    START_DATETIME = "{}".format(datetime.datetime.now())
    USE_DATA_DROPOUT = "use_data_dropout"

    OPTIMIZER_METHOD = "SGD"
    EPOCH = 26

    result_dict = None
    split_index = -1

    def __init__(self, filename):
        self.FILENAME = filename

    def save_result(self):
        """실험 결과를 파일에 적는다."""
            
        with open("result.csv", "a") as f:
            members = {}

            for attr in dir(self):
                value = getattr(self, attr)
                if not callable(value) and not attr.startswith("__"):
                    if isinstance(value, TrainResult):
                        members[attr] = value.to_dict()
                    else:
                        members[attr] = value

            print(members)
            f.write(json.dumps(members))
            f.write("\n")
