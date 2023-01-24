import torch
from sklearn.metrics import f1_score
import matplotlib as plt
from tqdm import tqdm
from datetime import datetime
import numpy as np

class Trainer:
    def __init__(self, model, criterion, optimizer=None, train_dl=None, val_test_dl=None, test_dl=None,
                 cuda=True, device=None):  # The patience for early stopping
        self._model = model
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._test_dl = test_dl
        self._crit = criterion
        self._optim = optimizer
        self._device = device 
            
    def save_checkpoint(self, epoch):
        torch.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = torch.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        torch.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
    def train_step(self,inp_data, labels):
        outputs = self._model(inp_data)
        loss = self._crit(outputs, labels)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return outputs, loss

    def val_test_step(self, inp_data, labels):
        outputs = self._model(inp_data)
        val_loss = self._crit(outputs, labels)
        predicted = outputs > 0.5
        return predicted, val_loss

    def train_epoch(self):
        self._model.train()
        loss_step = []
        correct, total = 0, 0
        for (inp_data, labels) in self._train_dl:
            # Move the data to the GPU
            outputs, loss = self.train_step(inp_data, labels)
            with torch.no_grad():
                predicted = outputs > 0.5
                total += labels.size(0)
                correct += 0.5 * (predicted == labels).sum()
                loss_step.append(loss.item())
        # dont forget the means here
        loss_curr_epoch = np.mean(loss_step)
        train_acc = (100 * correct / total).cpu()

        return loss_curr_epoch, train_acc
    
    def val_test(self, device=None):
        self._model.eval()  # set model to evaluation mode
        correct = 0
        total = 0
        loss_step = []
        with torch.no_grad():
            for inp_data, labels in self._val_test_dl:
                total += labels.size(0)
                predicted, val_loss = self.val_test_step(inp_data, labels)
                correct += 0.5 * (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                loss_step.append(val_loss.item())
        val_acc = 100 * correct / total
        # dont forget to take the means here
        val_loss_epoch = torch.tensor(loss_step).mean().numpy()

        return val_acc, val_loss_epoch
        
    def fit(self, epochs):
        best_loss = 1000
        best_acc = 0
        now = datetime.now()
        date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
        dict_log = {"train_acc": [], "val_acc": [],
                    "train_loss": [], "val_loss": []}
        train_acc, _ = self.val_test()
        val_acc, _ = self.val_test()
        print(
            f'Init Accuracy of the model: Train:{train_acc:.3f} \t Val:{val_acc:3f}')
        pbar = tqdm(range(epochs))
        for epoch in pbar:

            train_loss, train_acc = self.train_epoch()
            val_acc, val_loss = self.val_test()

            # Print epoch results to screen
            msg = (f'Ep {epoch+1}/{epochs}: Accuracy : Train:{train_acc:.2f} \t Val:{val_acc:.2f} || Loss: Train {train_loss:.3f} \t Val {val_loss:.3f}')
            pbar.set_description(msg)
            # Track stats
            dict_log["train_acc"].append(train_acc)
            dict_log["val_acc"].append(val_acc)
            dict_log["train_loss"].append(train_loss)
            dict_log["val_loss"].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optim.state_dict(),
                    'loss': val_loss,
                }, f"best_loss{date_time}.pth")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optim.state_dict(),
                    'loss': val_loss,
                }, f"best_acc{date_time}.pth")
        
        checkpoint = torch.load("best_loss{date_time}.pth")
        self._model.load_state_dict(checkpoint['model_state_dict'])
        return dict_log["train_loss"], dict_log["val_loss"]

    def plot_stats(self, dict_log, modelname="", baseline=90, title=None):
        fontsize = 14
        plt.subplots_adjust(hspace=0.3)
        plt.subplot(2, 1, 1)

        x_axis = list(range(len(dict_log["val_acc"])))
        plt.plot(dict_log["train_acc"], label=f'{modelname} Train accuracy')
        plt.scatter(x_axis, dict_log["train_acc"])

        plt.plot(dict_log["val_acc"],
                label=f'{modelname} Validation accuracy')
        plt.scatter(x_axis, dict_log["val_acc"])

        plt.ylabel('Accuracy in %')
        plt.xlabel('Number of Epochs')
        plt.title("Accuracy over epochs", fontsize=fontsize)
        plt.axhline(y=baseline, color='red', label="Acceptable accuracy")
        plt.legend(fontsize=fontsize)

        plt.subplot(2, 1, 2)
        plt.plot(dict_log["train_loss"], label="Training")

        plt.scatter(x_axis, dict_log["train_loss"], )
        plt.plot(dict_log["val_loss"], label='Validation')
        plt.scatter(x_axis, dict_log["val_loss"])

        plt.ylabel('Loss value')
        plt.xlabel('Number of Epochs')
        plt.title("Loss over epochs", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        if title is not None:
            plt.savefig(title)

    def test_model(self, dl):


        self._model.eval()  # set model to evaluation mode
        correct = 0
        total = 0
        loss_step = []
        with torch.no_grad():
            for data, labels in dl:
                outputs = self._model(data)
                val_loss = self._crit(outputs, labels)
                predicted = outputs > 0.5
                total += labels.size(0)
                correct += 0.5 * (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                loss_step.append(val_loss.item())
        val_acc = 100 * correct / total
        # dont forget to take the means here
        val_loss_epoch = torch.tensor(loss_step).mean().numpy()

        print("Daatset: Average loss: {:.4f}, Accuracy: {:.2f}%".format(val_loss_epoch, val_acc))
      
