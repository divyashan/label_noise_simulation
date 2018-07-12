
import matplotlib.pyplot as plt
from keras.callbacks import Callback


def plot_data(X_train, y_train, y_train_tilde, X_val, y_val, X_test, y_test, delta, p):
	plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
	plt.title("Train | Delta = " + str(delta) + " P(Y = 0) = " + str(p))
	plt.savefig("./" + str(delta) + "_" + str(p) + "_true_train.png")
	plt.clf()

	plt.scatter(X_train[:,0], X_train[:,1], c=y_train_tilde)
	plt.title("Train | Delta = " + str(delta) + " P(Y = 0) = " + str(p))
	plt.savefig("./" + str(delta) + "_" + str(p) + "_noisy_train.png")
	plt.clf()

	plt.scatter(X_val[:,0], X_val[:,1], c=y_val)
	plt.title("Val | Delta = " + str(delta) + " P(Y = 0) = " + str(p))
	plt.savefig("./" + str(delta) + "_" + str(p) + "_val.png")
	plt.clf()

	plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
	plt.title("Test | Delta = " + str(delta) + " P(Y = 0) = " + str(p))
	plt.savefig("./" + str(delta) + "_" + str(p) + "_test.png")
	plt.clf()

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def graph_gradient_path(gradients):
	# Take in list of gradients and generate path across hyperparameter surface
	parameters = []