import json
import os
import pathlib
import re

import numpy as np
import pandas as pd
import torch.utils.data as data
#from memory_profiler import profile

from settings import DATAPATH, CONFIGPATH

#test

class Dataset(data.Dataset):
    """데이터를 읽어오는 Class."""

    def __init__(self, filename, split=0, fold=10, phase="train", use_data_dropout=False):
        """데이터 셋 초기화."""
        self.orig_file = pathlib.Path(filename)
        self.filename = self.orig_file.stem
        processed_datapath = os.path.expanduser(f"{DATAPATH}/{self.filename}.npy")

        if not os.path.exists(processed_datapath):
            self.convert_and_save(processed_datapath)

        self.data = np.load(processed_datapath).astype(np.float32)
        assert phase in ["train", "test"]
        
        
        # 하나는 label
        self.feature_size = self.data.shape[1] - 1
        test_start = (len(self.data) * split) // fold
        test_end = (len(self.data) * (split + 1)) // fold
        self.phase = phase

        self.train_data = np.concatenate(
            (self.data[:test_start], self.data[test_end:]), axis=0
        )
        self.test_data = self.data[test_start:test_end]
        self.data = self.train_data if phase == "train" else self.test_data
        self.use_data_dropout = use_data_dropout
        self.split = split

    def convert_and_save_simple(self, processed_datapath):
        # TODO: 메모리 적게 먹는 버젼 구현
        column_type = json.load(open(f"{CONFIGPATH}/column_list.json"))

        with open(self.orig_file) as f:
            header = f.readline().split(",")
            columns = []
            for _ in range(len(header)):
                columns.append([])
            for line in f:
                for idx, data in enumerate(line.split(",")):
                    columns[idx].append(data)
        for col in range(1, len(columns)):
            col_type = column_type[header[col]]
            columns[col] = self.gather_special_responses_simple(columns[col], col_type)

    def convert_and_save(self, processed_datapath):
        """변환한 데이터를 파일로 저장."""
        datapath = self.orig_file
        column_type = json.load(open(f"{CONFIGPATH}/column_list.json"))

        df = pd.read_csv(datapath, sep=",", dtype=column_type)
        df, column_names = self.convert_dataset(df, column_type)

        np.random.shuffle(df)
        np.save(processed_datapath, df)
        np.save(processed_datapath.replace(".npy", "_columnnames.npy"), column_names)
        print("Saved at %s" % processed_datapath)

    def to_csv_file(self):
        # csv 파일로 데이터 저장
        with open(f"{DATAPATH}/parsed_{self.phase}.csv", "w") as f:
            for line in self.data:
                f.write(",".join([str(d) for d in line]))
                f.write("\n")

    def gather_special_responses_simple(self, col, c_type):
        # TODO: 구현
        is_categorical = c_type == "object"
        if not is_categorical:
            col = [float(c) for c in col]

    def is_categorical(self, c_type, colname):
        if colname in ["BS3_3", "BE3_32", "D_2_wk"]:
            return False
        if c_type == "object":
            return True
        return False

    # @profile
    def gather_special_responses(self, col, c_type, colname):
        is_categorical = self.is_categorical(c_type, colname)
        return_list = []
        empty = pd.isnull(col)
        # print(col[-100:])
        #if empty.any():
        return_list.append(empty.astype(float))

        if is_categorical:
            col[empty] = "-9999"
            unique_values = np.unique(col)
            mapping = {v: i for i, v in enumerate(unique_values)}
            print("%d unique values - %s" % (len(unique_values), str(unique_values)))
            new_col = [mapping[v] for v in col]
            #one_hot_size = 16  # for simplicity, make the size of one hot vector always 16.
            one_hot_size = len(unique_values)
            one_hot = np.eye(one_hot_size)[new_col]
            #col = np.transpose(one_hot)
            col = one_hot * (1.0 - empty[:, None].astype(float))  # zero out the one-hot vector if empty
            print(col.shape, empty.shape)
            return np.concatenate([empty[:, None].astype(float), col], axis=1)
        else:
            col = col.astype(np.float32)
            col[empty] = 0.0
            max_val = np.nanmax(col)

            special_values = [8.0, 88.0, 888.0, 8888.0, 88888.0, 888888.0]
            is_special = None
            if max_val in special_values:
                # special value 처리
                # 일단 88, 888 처럼 두가지 이상의 패턴이 있는지 본다
                idx = special_values.index(max_val)
                max_small = special_values[idx - 1]
                is_special = col == max_val

                # value_nan = False
                # 이전 88 패턴이 존재하고 그 사이에 값이 있으면 아마도 이전 88은 그냥 값일 가능성이 높음

                value_between = np.any(np.logical_and(max_small < col, col < max_val))
                value_previous = np.any(max_small == col)
                # print(max_small == col)
                # for value in col:
                #     if max_small < value and value < max_val:
                #         value_between = value
                #     if max_small == value:
                #         value_previous = True

                if value_previous:
                    if not value_between:
                        is_special = np.logical_or(is_special, col == max_small)

                col[is_special] = 0.0

                return_list.append(is_special.astype(float))

            # empty 값도 평균시에서 제거 할지?
            if np.any(is_special):
                has_ordinary_value = np.logical_not(np.logical_or(is_special, empty))
                #has_ordinary_value = np.logical_not(is_special)
            else:
                has_ordinary_value = np.logical_not(empty)
                #has_ordinary_value = [True] * len(col)

            ordinary_values = col[has_ordinary_value]
            mean = ordinary_values.mean()
            std = ordinary_values.std()
            # print(has_ordinary_value[-100:])
            #print(len(ordinary_values), mean, std)
            # print(col[-100:])

            normalized = (col - mean) / std
            col[has_ordinary_value] = normalized[has_ordinary_value]
            # print(col[-100:])
            return_list.append(col)
            #print(round(col.nbytes / 1024 / 1024, 2))
            return np.stack(return_list, axis=1)

        return return_list

    #@profile
    def convert_dataset(self, dataframe, column_type):
        all_column_names = []
        all_columns = []
        unknown_column = []
        max_num_columns = 0
        for i, column_name in enumerate(dataframe.columns):
            #if i % 10 > 0:
            #    continue
            try:
                c_type = column_type[column_name]
            except KeyError:
                unknown_column.append(column_name)
                continue
            # 데이터 값이 없는 경우 -1.0으로 초기화 한다.
            col = dataframe.values[:, i]

            if i == 0:
                # the first column -> label
                all_columns.append(col[:, None])
                continue

            col[col == ""] = np.nan
            col[col == " "] = np.nan

            # 맨처음 AUD / BS6_2_2 컬럼 이후로부터는 categorical 데이터
            col = self.gather_special_responses(col, c_type, column_name)
            print(col.shape)
            #col = np.stack(col, axis=1)
            colsize = col.shape[1]
            col = np.concatenate([col, np.zeros((col.shape[0], 25 - colsize), dtype=col.dtype)], axis=1)
            print(column_name, col.shape)
            # input()
            all_columns.append(col)
            all_column_names += [column_name] * col.shape[1]
            max_num_columns = max(max_num_columns, col.shape[1])
        if unknown_column:
            raise KeyError(",".join(unknown_column))
        all_columns = np.concatenate(all_columns, axis=-1).astype(np.float32)
        print("Final dimension is %s" % str(all_columns.shape))
        print("max num columns is %d" % max_num_columns)
        return all_columns, all_column_names

    def __getitem__(self, index):
        row = self.data[index].copy()
        # Target 값인 AUD 는 column의 맨 처음
        y = row[0]
        x = row[1:]

        if self.use_data_dropout:
            num_variables = x.shape[0] // 25
            x = x.reshape((num_variables, 25))
            #dropout_indices = np.random.randint(num_variables)
            dropout_indices = np.random.choice(np.arange(num_variables), size=(int(num_variables // 8)),
                                               replace=False)
            x[dropout_indices, :1] = 1.0  # first entry is for "emptiness"
            x[dropout_indices, 1:0] = 0.0
            #x = x * num_variables / (num_variables - 1)  # scale the vector magnitudes to account for the dropout
            x = x.flatten()
        
        return x, y

    def __len__(self):
        return len(self.data)


def to_space_to_empty(filename: pathlib):
    """comma 다음에 space 존재하는 부분을 제거한다."""
    space = re.compile(r", ")
    new_filename = f"{DATAPATH}/{filename.stem}_no_space.csv"
    with open(new_filename, "w") as w:
        with open(filename) as f:
            for line in f:
                w.write(space.sub(",", line))
    return new_filename


def describe(filename):
    """데이터 셋의 통계 정보를 출력한다."""
    column_type = json.load(open(f"{CONFIGPATH}/column_list.json"))
    column_list = [c for c in column_type]
    datapath = os.path.expanduser(f"{DATAPATH}/{filename}.csv")

    # df = pd.read_csv(datapath, sep=",")
    df = pd.read_csv(
        datapath, sep=",", low_memory=False, dtype=column_type, usecols=column_list
    )
    # print(df.dtypes)

    df.describe(include="all").to_csv(f"{DATAPATH}/my_description.csv")


def run(filename):
    """데이터셋 분석."""
    filename = pathlib.Path(filename)
    filename = to_space_to_empty(filename)

    split = 0

    train_dataset = Dataset(split=split, fold=10, phase="train", filename=filename)
    test_dataset = Dataset(split=split, fold=10, phase="test", filename=filename)

    train_dataset.to_csv_file()
    test_dataset.to_csv_file()
