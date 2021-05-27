import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from model import ResNet9
from bayesian_optimization import BayesianOptimizer


def show_batch(images):
    """
    Prints one batch of images in a single image.

    Parameters
    ----------
    images:
        Batch of images in training data from the DataLoader.
    """
    images = torchvision.utils.make_grid(images).numpy()
    plt.imshow(np.transpose(images, (1, 2, 0)))
    plt.show()


def accuracy(pred, true):
    """
    Calculates the accuracy given a tensor of probability predictions for each class and a Tensor of ground truth
    class indexes. Note that for both the first dimension defines the batch size.

    Parameters
    ----------
    pred: torch.Tensor
        Tensor of probability predictions for each class.

    true: torch.Tensor
        Tensor of indexes of the ground truth classes.

    Returns
    -------
    Accuracy of predictions.

    """
    class_index_pred = pred.detach().numpy().argmax(axis=1)
    return np.sum(true.detach().numpy() == class_index_pred) / len(pred)


def run_one_epoch(lr, train_l, val_l, seed):
    """
    Runs one epoch of training using the specified learning rate lr and returns the negative average
    validation loss.

    Parameters
    ----------
    lr: float
        Learning rate of the model.

    train_l: DataLoader
        Torch's DataLoaders constituting an iterator over the training dataset.

    val_l: DataLoader
        Torch's DataLoaders constituting an iterator over the validation dataset.

    seed: int
        Seed for Numpy and Torch.

    Returns
    -------
    Tuple containing the negative validation loss and the model trained on the specified learning rate.
    """
    # set the seed to initialize same model and randomness on all epochs to allow fair comparison
    np.random.seed(seed)
    torch.manual_seed(seed)

    # get our model and define the optimizer as well as the loss criterion
    model = ResNet9(in_channels=1, num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loop = tqdm(train_l)  # tqdm wrapper used to print progress bar
    for data in train_loop:
        # unpack images and labels
        images, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # calculate and apply the gradients
        loss.backward()
        optimizer.step()

        # print some training information
        train_loop.set_postfix({"Loss": loss.item(), "Accuracy": accuracy(outputs, labels)})

    # let's validate our model
    print("Validating ...")
    with torch.no_grad():
        cum_val_loss = 0.0
        cum_acc = 0.0
        for data in val_l:
            # unpack images and labels
            images, labels = data

            # calculate loss
            outputs = model(images)
            cum_val_loss += criterion(outputs, labels)
            cum_acc += accuracy(outputs, labels)

        # print some validation information
        avg_val_loss = cum_val_loss / len(val_loader)
        avg_val_acc = cum_acc / len(val_loader)
        print("---> Validation-Loss: %.4f & Validation-Accuracy: %.4f" % (avg_val_loss, avg_val_acc))

    return -avg_val_loss, model


if __name__ == "__main__":

    # --- 1. setup some parameters ---
    path_to_model = "./ResNet9_model"
    batch_size = 32
    n_iterations = 10
    seed = 94
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- 2. let's get our dataset ---
    # define some transformations we want to apply to our images
    transform = transforms.Compose(
        [transforms.ToTensor()])

    # create a dataset and dataloader for training and validation
    train_ds = torchvision.datasets.KMNIST(root='./data', train=True,
                                           download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    val_ds = torchvision.datasets.KMNIST(root='./data', train=False,
                                         download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # --- 3. let's start training and evaluation ---

    # define our kernel and GP
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=10)

    # define our Bayesian Optimizer and pass the objective function
    bo = BayesianOptimizer(lambda x: run_one_epoch(x, train_loader, val_loader, seed),
                           gp,
                           mode="linear",
                           bound=[1e-5, 1.0])

    # run the evaluation
    found_lr, best_model = bo.eval(n_iter=n_iterations)
    print("\n--> Found learning-rate after %d iterations: %f" % (n_iterations, found_lr))

    # save all the plots
    bo.save_all_plots()

    # save the best model
    torch.save(best_model.state_dict(), path_to_model)
