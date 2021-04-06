import torch


def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_classifier_model(model_name, feature_size, nchs, act):
    model = None
    if model_name == "Classifier":
        model = Classifier(feature_size, nchs).cuda()
    elif model_name == "ClassifierWithDropout":
        model = ClassifierWithDropout(feature_size, act, nchs).cuda()
    elif model_name == "ClassifierWithBatchNorm":
        model = ClassifierWithBatchNorm(feature_size, act, nchs).cuda()
    elif model_name == "ClassifierWithEmbedding":
        model = ClassifierWithEmbedding(feature_size, act, nchs).cuda()
    elif model_name == "ClassifierWithAttention":
        model = ClassifierWithAttention(feature_size, act, nchs).cuda()
    else:
        raise ValueError("Unknown model {}".format(model_name))

    model.apply(init_weights)
    return model


class Classifier(torch.nn.Sequential):
    """ Classification network using |nchs| number of nodes in each layer"""

    def __init__(self, inch, nchs_orig: list):
        super().__init__()
        nchs = [inch] + nchs_orig

        for i in range(len(nchs) - 1):
            self.add_module("Linear%d" % i, torch.nn.Linear(nchs[i], nchs[i + 1]))
            if i < len(nchs) - 2:
                # self.add_module("ReLU%d" % i, torch.nn.LeakyReLU())
                # self.add_module("Dropout%d" % i, torch.nn.Dropout(p=0.5))
                self.add_module("Tanh%d" % i, torch.nn.Tanh())
                # self.add_module("sigmoid%d" % i, torch.nn.Sigmoid())


class ClassifierWithDropout(torch.nn.Sequential):
    """ Classification network using |nchs| number of nodes in each layer"""

    def __init__(self, inch, act, nchs_orig: list):
        super().__init__()
        nchs = [inch] + nchs_orig

        for i in range(len(nchs) - 1):
            self.add_module("Linear%d" % i, torch.nn.Linear(nchs[i], nchs[i + 1]))
            if i < len(nchs) - 2:
                # self.add_module("ReLU%d" % i, torch.nn.LeakyReLU())
                self.add_module("Dropout%d" % i, torch.nn.Dropout(p=0.5))
                self.add_module("Activation%d" % i, get_activation(act))
                # self.add_module("sigmoid%d" % i, torch.nn.Sigmoid())


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def get_activation(type):
    if type == "tanh":
        return torch.nn.Tanh()
    elif type == "ReLU":
        return torch.nn.ReLU()
    elif type == "LReLU":
        return torch.nn.LeakyReLU()
    elif type == "swish":
        return Swish()
    else:
        raise ValueError(type)

class ClassifierWithBatchNorm(torch.nn.Sequential):
    """ Classification network using |nchs| number of nodes in each layer"""

    def __init__(self, inch, act, nchs_orig: list):
        super().__init__()
        nchs = [inch] + nchs_orig

        for i in range(len(nchs) - 1):
            self.add_module("Linear%d" % i, torch.nn.Linear(nchs[i], nchs[i + 1]))
            if i < len(nchs) - 2:
                # self.add_module("ReLU%d" % i, torch.nn.LeakyReLU())
                # self.add_module("Dropout%d" % i, torch.nn.Dropout(p=0.5))
                self.add_module("BatchNorm%d" % i, torch.nn.BatchNorm1d(nchs[i + 1]))
                self.add_module("Activation%d" % i, get_activation(act))
                self.add_module("Dropout%d" % i, torch.nn.Dropout(p=0.5))
                # self.add_module("sigmoid%d" % i, torch.nn.Sigmoid())

class ClassifierWithEmbedding(torch.nn.Module):
    def __init__(self, inch, act, nchs_orig: list):
        super().__init__()
        assert inch % 25 == 0, "Every variable must have the same dimension (17)"
        embd_ch = (inch // 25) * 8
        self.projector = torch.nn.Sequential(
            torch.nn.Conv1d(inch, embd_ch, 1, groups=inch // 25),
            get_activation(act)
        )
        self.classifier = ClassifierWithBatchNorm(embd_ch, act, nchs_orig)
        #self.classifier = ClassifierWithDropout(embd_ch, act, nchs_orig)

    def forward(self, x):
        embedding = self.projector(x[:, :, None])[:, :, 0]
        embedding = embedding.view(embedding.shape[0], -1, 8)
        embedding = embedding / ((embedding ** 2).mean(dim=1, keepdim=True).sqrt() + 1e-7)
        embedding = embedding.view(embedding.shape[0], -1)
        return self.classifier(embedding)

class ClassifierWithAttention(torch.nn.Module):
    def __init__(self, inch, act, nchs_orig: list):
        super().__init__()
        assert inch % 25 == 0, "Every variable must have the same dimension (17)"
        embd_ch = (inch // 25) * 16
        self.projector = torch.nn.Sequential(
            torch.nn.Conv1d(inch, embd_ch, 1, groups=inch // 25),
            get_activation(act)
        )
        self.classifier = ClassifierWithBatchNorm(embd_ch // 2 + (inch // 25) ** 2, act, nchs_orig)
        #self.classifier = ClassifierWithDropout(embd_ch, act, nchs_orig)

    def forward(self, x):
        bs = x.shape[0]
        embedding = self.projector(x[:, :, None])[:, :, 0]
        embedding = embedding.view(bs, -1, 16)
        embedding = embedding / ((embedding ** 2).mean(dim=1, keepdim=True).sqrt() + 1e-7)

        embedding_standalone = embedding[:, :, :8]
        embedding_for_attention = embedding[:, :, 8:]

        attention_features = torch.bmm(embedding_for_attention, embedding_for_attention.transpose(1, 2))
        
        embedding = torch.cat([embedding_standalone.flatten(1), attention_features.flatten(1)], dim=1)
        #embedding = embedding.view(embedding.shape[0], -1)
        return self.classifier(embedding)


        

