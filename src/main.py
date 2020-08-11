import logging

from absl import app
from absl import flags

import dataset
import train

FLAGS = flags.FLAGS

flags.DEFINE_boolean("dataset", False, "데이터 분석")
flags.DEFINE_boolean("train", False, "모델 학습")
flags.DEFINE_string("options", None, "학습 옵션 파일 위치")
flags.DEFINE_string("filename", "../datasets/medical_data_6.csv", "파일 위치")

def main(_):
    if not FLAGS.dataset and not FLAGS.train:
        logging.info("dataset 이나 train 옵션 중 하나는 True 여야 합니다.")

    if FLAGS.dataset:
        # 데이터셋 분석
        dataset.run(FLAGS.filename)

    if FLAGS.train:
        # 학습
        train.run(FLAGS.filename)

    return 0


if __name__ == "__main__":
    app.run(main)
