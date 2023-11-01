
import matplotlib.pyplot as plt
def plot_gan_loss(G_losses, D_losses, file_path):
    """ Plot the loss of the generator and discriminator during training

    Args:
        G_losses (list): list of generator losses
        D_losses (list): list of discriminator losses
        file_path (str): path to save the plot
    """
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_path)
    plt.show()
    
    
def plot_D_accuracy(D_accuracy, file_path):
    ''' Plot the discriminator accuracy during training

    Args:
        D_accuracy: list of discriminator accuracy
        file_path: path to save the plot
    '''
    # discriminator accuracy
    plt.figure(figsize=(10,5))
    plt.title("Discriminator Accuracy During Training")
    plt.plot(D_accuracy,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(file_path)
    plt.show()

