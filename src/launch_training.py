import os

LR = [0.5, 0.1, 0.02, 0.004]
optimizer = ["Adadelta", "SGD"]
batch_size = [2, 16, 64, 256]

for lr in LR[3:]:
    for opt in optimizer:
        for bs in batch_size:
            name = "bs%d_lr%f_opt%s" % (bs, lr, opt)
            command = "python train.py --name %s --init_lr %f --batch_size %d --optimizer %s" % (name, lr, bs, opt)
            os.system(command)
