import torch as tr
import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
from matplotlib import pyplot as plt
import torchmetrics
import pandas as pd


batch_size = 32
output_size = 2
hidden_size = 64        # to experiment with
run_recurrent = False    # else run Token-wise MLP
use_RNN = False          # otherwise GRU
atten_size = 0          # atten > 0 means using restricted self atten
RELU, SIGMOID, TANH = "RelU", "Sigmoid", "Tanh"

reload_model = False
num_epochs = 10
learning_rate = 0.0004
test_interval = 50
train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x,self.matrix)
        if self.use_bias:
            x = x+ self.bias
        return x

# Implements RNN Unit
class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation_name = None):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid
        self.activation_name = activation_name
        if self.activation_name is None:
            self.activation_name = "no_activation"

        # RNN Cell weights
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # what else?
        self.activation = None
        if activation_name == RELU:
            self.activation = nn.ReLU()
        elif activation_name == SIGMOID:
            self.activation = nn.Sigmoid()
        elif activation_name == TANH:
            self.activation = nn.Tanh()
        self.out = nn.Linear(hidden_size, output_size)

    def name(self):
        return "RNN" + " " + self.activation_name

    def forward(self, x, hidden_state_):
        concat = torch.cat((x, hidden_state_), 1)
        hidden = self.in2hidden(concat)
        if self.activation is not None:
            hidden = self.activation(hidden)
        out = self.sigmoid(self.out(hidden))
        return out, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


# Implements GRU Unit
class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, activation_name=None):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.activation_name = activation_name
        if self.activation_name is None:
            self.activation_name = "no_activation"
        # GRU Cell weights
        # self.something =
        # etc ...
        self.in2hidden_update = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2hidden_reset = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_layer = nn.Linear(input_size + hidden_size, hidden_size)
        self.activation = None
        if activation_name == RELU:
            self.activation = nn.functional.relu
        elif activation_name == SIGMOID:
            self.activation = nn.functional.sigmoid
        elif activation_name == TANH:
            self.activation = torch.tanh
        self.sigmoid = nn.functional.sigmoid
        self.out = nn.Linear(hidden_size, output_size)



    def name(self):
        return "GRU" + " " + self.activation_name

    def forward(self, x, hidden_state_):
        # Implementation of GRU cell
        concat_inputs = torch.cat((x, hidden_state_), 1)
        update_gate = self.sigmoid(self.in2hidden_update(concat_inputs))
        reset_gate = self.sigmoid(self.in2hidden_reset(concat_inputs))

        # missing implementation
        concat_hidden_reset_x = torch.cat((x, reset_gate * hidden_state_), 1)
        hidden_tilda = self.hidden_layer(concat_hidden_reset_x)
        if self.activation is not None:
            hidden_tilda = self.activation(hidden_tilda)
        hidden = torch.multiply(1 - update_gate, hidden_state_) + torch.multiply(update_gate, hidden_tilda)

        out = self.sigmoid(self.out(hidden))
        return out, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.ReLU = torch.nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        # Token-wise MLP network weights
        self.layer1 = MatMul(input_size,hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.layer2 = nn.Linear(hidden_size, 48)
        self.layer3 = nn.Linear(48, 24)
        self.layer4 = MatMul(24, 12)
        self.act4 = torch.nn.ReLU()
        self.layer5 = MatMul(12, 8)
        self.out = nn.Linear(8, output_size)



    def name(self):
        return "MLP, 4 layers, dropout = 0.1"

    def forward(self, x):

        # Token-wise MLP network implementation

        x = self.layer1(x)
        x = self.ReLU(x)
        # rest
        x = self.ReLU(self.layer2(x))
        x = self.dropout(self.layer3(x))
        x = self.layer4(x)
        x = self.act4(x)
        x = self.layer5(x)

        # out
        x = self.out(x)
        return x


class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(2)

        # Token-wise MLP + Restricted Attention network implementation

        self.layer1 = MatMul(input_size,hidden_size)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        # rest ...
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        self.out = ExMLP(hidden_size, output_size, hidden_size)


    def name(self):
        return "MLP_atten"

    def forward(self, x):

        # Token-wise MLP + Restricted Attention network implementation

        x = self.layer1(x) # (batch_size, num_days, values) --> (batch_size, num_days, hidden_size)
        x = self.ReLU(x)


        # generating x in offsets between -atten_size and atten_size
        # with zero padding at the ends
        padded = pad(x,(0,0,atten_size,atten_size,0,0))

        x_nei = []
        for k in range(-atten_size,atten_size+1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei,2)
        x_nei = x_nei[:,atten_size:-atten_size,:]  # (batch, num_days, 2 * attention_size + 1, values)

        # x_nei has an additional axis that corresponds to the offset

        # Applying attention layer

        query = self.W_q(x)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)

        dot_product = torch.einsum('lij, litj -> lit', query, keys) / self.sqrt_hidden_size
        atten_weights = self.softmax(dot_product)
        val_out = torch.einsum('lij, lijt -> lit', atten_weights, vals)
        x = self.out(x + val_out)
        return x, atten_weights


# prints portion of the review (20-30 first words), with the sub-scores each work obtained
# prints also the final scores, the softmaxed prediction values and the true label values
def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    labels_to_sentiment = lambda l1, l2: 'Positive' if l1 > l2 else 'Negative'
    print('-' * 120)
    sentence = ''

    words_len = min(len(rev_text), 20)
    for counter,text in enumerate(rev_text):
        if counter > words_len:
            break
        sentence += ' ' + text
        if counter % 7 == 0 and counter > 0:
            sentence += '\n' + ' ' * 9

    print(f"Sentence:{sentence}")
    print("Scores: ")
    data_scores = pd.DataFrame({"Word": rev_text[:words_len], "Positive": sbs1[:words_len], "Negative": sbs2[:words_len]})
    print(data_scores.to_string(index=False))

    sbs1_score, sbs2_score = sbs1.mean(), sbs2.mean()
    print(f"\nPositive Score: {sbs1_score}\nNegative Score: {sbs2_score}\n")

    original_sentiment = labels_to_sentiment(lbl1, lbl2)
    print(f"Sentence Sentiment: {original_sentiment}")
    predicted_sentiment = labels_to_sentiment(sbs1_score, sbs2_score)
    print(f"Prediction Sentiment: {predicted_sentiment}")
    print(f"\nPredicted {'right!!!' if original_sentiment == predicted_sentiment else 'wrong'}")
    print('-' * 120)

if run_recurrent:
    if use_RNN:
        model = ExRNN(input_size, output_size, hidden_size)
    else:
        model = ExGRU(input_size, output_size, hidden_size, TANH)
else:
    if atten_size > 0:
        model = ExLRestSelfAtten(input_size, output_size, hidden_size)
    else:
        model = ExMLP(input_size, output_size, hidden_size)


# import wandb
# wandb.init(project=f"Deep Learning - EX2, {model.name()}", entity="nadav-alali")
# wandb.config = {
#   "learning_rate": learning_rate,
#   "epochs": num_epochs,
#   "batch_size": batch_size
# }


print("Using model: " + model.name())

if reload_model:
    print("Reloading model")
    model.load_state_dict(torch.load(model.name() + ".pth")).state_dict()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = 1.0
test_loss = 1.0

# training steps in which a test step is executed every test_interval
accuracy = torchmetrics.Accuracy()
roc_accuracy = torchmetrics.AUROC(num_classes=2)
F1_accuracy = torchmetrics.F1()
train_acc, test_acc = [], []
train_roc, test_roc = 0.0, 0.0
train_f1, test_f1 = 0.0, 0.0
for epoch in range(num_epochs):
    train_acc_epoch, test_acc_epoch = 0.0, 0.0
    train_roc_epoch, test_roc_epoch = 0.0, 0.0
    train_f1_epoch, test_f1_epoch = 0.0, 0.0

    itr = 0  # iteration counter within each epoch
    train_iteration = 0
    test_iteration = 0

    for labels, reviews, reviews_text in train_dataset:   # getting training batches

        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch
        else:
            test_iter = False

        # Recurrent nets (RNN/GRU)

        if run_recurrent:
            hidden_state = model.init_hidden(int(labels.shape[0]))

            for i in range(num_words):
                output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE


        else:

        # Token-wise networks (MLP / MLP + Atten.)

            sub_score = []
            if atten_size > 0:
                # MLP + atten
                sub_score, atten_weights = model(reviews)
            else:
                # MLP
                sub_score = model(reviews)

            output = torch.mean(sub_score, 1)

        # cross-entropy loss
        # output = nn.Sigmoid()(output)

        loss = criterion(output, labels)

        # optimize in training iterations

        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses
        int_labels = labels.int()
        acc = accuracy(output, int_labels)
        roc = roc_accuracy(output, int_labels)
        f1 = F1_accuracy(output, int_labels)
        # acc = accuracy_score(labels, 1 * (output > 0.5))
        if test_iter:
            test_acc_epoch += acc
            test_roc_epoch += roc
            test_f1_epoch += f1
            test_iteration += 1
            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss
        else:
            train_acc_epoch += acc
            train_roc_epoch += roc
            train_f1_epoch += f1
            train_iteration += 1
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

        if test_iter:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{itr + 1}/{len(train_dataset)}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}"
            )

            if not run_recurrent:
                nump_subs = sub_score.detach().numpy()
                labels = labels.detach().numpy()
                print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

            # saving the model
            torch.save(model, model.name() + ".pth")
    train_acc.append(train_acc_epoch / train_iteration)
    test_acc.append(test_acc_epoch / test_iteration)
    train_roc, test_roc = train_roc_epoch / train_iteration, test_roc_epoch / test_iteration
    train_f1, test_f1 = train_f1_epoch / train_iteration, test_f1_epoch / test_iteration
    # wandb.log({"test accuracy": test_acc[-1], "train accuracy": train_acc[-1]})

# wandb.watch(model)
print(f"Model accuracy: train = {train_acc[-1]}, test = {test_acc[-1]}")
print(f"Model ROC AUC: train = {train_roc}, test = {test_roc}")
print(f"Model F1 accuracy: train = {train_f1}, test = {test_f1}")


plt.title(f'{model.name()}, hidden_size={hidden_size}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(train_acc, label=f'train acc')
plt.plot(test_acc, label=f'test acc')
plt.legend()
plt.savefig(f"ACC_{model.name()}_bs_{batch_size}_hs_{hidden_size}.png")
plt.show()


my_test_texts = []
my_test_texts.append(" this movie is very very bad ,the worst movie ")
my_test_texts.append(" this movie is so great")
my_test_texts.append("I really  liked the fish and animations the anther casting was not so good ")
my_test_texts.append("Calling this movie very very good and great is a lie and it's actually the worst movie ever")
my_test_texts.append("This movie is great said no one")
my_test_labels = ["negative", "positive", "positive", "negative", "negative"]

if not run_recurrent:
    my_data = pd.DataFrame({"review": my_test_texts, "sentiment": my_test_labels})
    my_data= my_data.reset_index(drop=True)
    text_test = ld.ReviewDataset(my_data["review"],my_data["sentiment"])
    dataloader = ld.DataLoader(text_test, batch_size=len(my_test_texts),
                                      shuffle=False, collate_fn=ld.collact_batch)

    for labels, reviews, reviews_text in dataloader:
        sub_score = []
        if atten_size > 0:
            # MLP + atten
            sub_score, atten_weights = model(reviews)
        else:
            # MLP
            sub_score = model(reviews)
        output = torch.mean(sub_score, 1)
        nump_subs = sub_score.detach().numpy()
        labels = labels.detach().numpy()
        for i in range(len(my_test_texts)):
            print_review(reviews_text[i], nump_subs[i,:,0], nump_subs[i,:,1], labels[i,0], labels[i,1])

