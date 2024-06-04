import torch
import copy


def get_data_loaders(df, target, val_rate=.2, batch_size=64):
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(df.drop(target, axis=1), df[target], test_size=val_rate)

    training_set = torch.utils.data.TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32))

    validation_set = torch.utils.data.TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32))

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True) # here shuffle is recommended for using early stopping on big batches later
    return training_loader, validation_loader

# training_loader, validation_loader = get_data_loaders(df, 'churn')

# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, n_features_in : int):
#         super().__init__()
#         self.linear_relu_stack = torch.nn.Sequential(
#             torch.nn.Linear(n_features_in, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 1),
#             torch.nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.linear_relu_stack(x)
#         return x

# model = NeuralNetwork(df.shape[1]-1)

# loss_fn = torch.nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

class Trainer():
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 training_loader,
                 validation_loader,
                 *,
                 bbatch_size : int | None = 16,
                 val_bbatch_size : int | None = 16,
                 print_loss : str | None = None,
                 graph_loss : str | None = 'bbatch',
                 early_stopping : str | None = 'bbatch',
                 early_stopping_params : dict | None = None,
                 max_epochs : int = 100) -> None:
        '''
        Loss tracking is done by storing the losses as lists. 2 kinds for epoch and bbatch, 2 kinds for train and loss - 4 in total. 
        Validation loss (epoch or bbatch) is calculated only if needed.

        Loss tracking for user is done by 1) printing 2) graphing 3) not done.

        Loss tracking is done either on big batches or epochs.

        Early stopping is done either on big batches or epochs.

        Optional args:
            bbatch_size: size of a big batch (in usual batches, defined by training loader) (16)
            val_bbatch_size: same for validation. Losses for big val batches will be calculated after a big train batch has run

            print_loss: print loss (None) ['epoch', 'bbatch', None]
            graph_loss: graph loss ('bbatch') ['epoch', 'bbatch', None]
                Not recommended to use both

            early_stopping: early stopping done on epochs, big batches or not implemented ('bbatch') ['epoch', 'bbatch', None]
                It is recommended that early stopping matches loss reporting (big batches/epochs)
            early_stopping_params: dict
                X
                path: path to save the model ('models/model' is default and recommended)
                save: save mode for early stopping ('stop') ['new', 'overwrite', 'stop']
                    'new' - save new model each time improvement is noticed (not recommended),
                    'overwrite' - overwrite model each time improvement is noticed (recommended for smaller NNs)
                    'stop' - do not save at all, simply break training if early stop is triggered (recommended for larger NNs)

            max_epochs: maximum number of epochs to run (100)
        '''
        #
        # option validation
        #


        #
        # option processing
        #
        if early_stopping_params is None:
            early_stopping_params = {}

        early_stopping_params.setdefault('path', 'models/model')
        early_stopping_params.setdefault('save', 'stop')

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.bbatch_size = bbatch_size
        self.val_bbatch_size = val_bbatch_size
        self.print_loss = print_loss
        self.graph_loss = graph_loss
        self.early_stopping = early_stopping
        self.early_stopping_params = early_stopping_params
        self.max_epochs = max_epochs

        self.cur_epoch = None
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.train_bbatch_losses = []
        self.val_bbatch_losses = []
        return None


    def train(self) -> None:
        if self.print_loss == 'bbatch':
            print(f'|{"epoch":^12}|{"bbatch":^12}|{"train":^12}|{"val":^12}|')

        for epoch in range(self.max_epochs):
            self.cur_epoch = epoch

            train_loss = self._train_epoch()
            self.train_epoch_losses.append(train_loss)

            if self.print_loss == 'epoch' or self.graph_loss == 'epoch' or self.early_stopping == 'epoch':
                val_loss = self._get_val_loss(on='epoch')

            if self.print_loss == 'epoch':
                print(f'|{self.cur_epoch+1:^12}|{"-":^12}|{train_loss:^12.4f}|{val_loss:^12.4f}|')

            if self.graph_loss == 'epoch':
                pass

            if self.early_stopping == 'epoch':
                pass
                # if early stop has activated: stop training and print it out

        # if all epochs have run, print that
        return None


    def _train_epoch(self) -> float:
        '''
        Train for epoch, return mean training loss
        '''
        epoch_loss = 0.
        prev_value_epoch_loss = 0. # for big training batch loss reporting

        for batch_index, (inputs, labels) in enumerate(self.training_loader):

            # training stuff
            self.model.train()
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs, labels.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            # sum loss for reporting
            epoch_loss += loss.item()

            if self.bbatch_size is not None and batch_index % self.bbatch_size == self.bbatch_size - 1:

                # we use this to extract only the loss attributed to this big batch
                # we do not track big batch explicitly, because we do that for epoch loss
                training_bbatch_loss = (epoch_loss - prev_value_epoch_loss) / self.bbatch_size
                prev_value_epoch_loss = copy.deepcopy(epoch_loss)
                self.train_bbatch_losses.append(training_bbatch_loss)

                if self.print_loss == 'bbatch' or self.graph_loss == 'bbatch' or self.early_stopping == 'bbatch':
                    val_bbatch_loss = self._get_val_loss(on='bbatch')
                    self.val_bbatch_losses.append(val_bbatch_loss)

                if self.print_loss == 'bbatch':
                    print(f'|{self.cur_epoch+1:^12}|{batch_index+1:^12}|{training_bbatch_loss:^12.4f}|{val_bbatch_loss:^12.4f}|')

                if self.graph_loss == 'bbatch':
                    pass

                if self.early_stopping == 'bbatch':
                    pass
                    # if early stop has activated: stop training and print it out

        epoch_loss = epoch_loss / (batch_index + 1)
        return epoch_loss


    def _get_val_loss(self, on : str) -> float:
        '''
        Function sets the modes (eval, train) itself
        '''
        val_loss = 0.
        self.model.eval()

        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(self.validation_loader):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.unsqueeze(1))
                val_loss += loss.item()

                if on == 'bbatch' and batch_index % self.val_bbatch_size == self.val_bbatch_size - 1:
                    break
        
        val_loss = val_loss / (batch_index + 1)
        return val_loss


# trainer = Trainer(model, loss_fn, optimizer, training_loader, validation_loader, max_epochs=20)
# trainer.train()