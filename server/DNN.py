#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LeakyReLU, SimpleRNN, Reshape
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Conv2D
from sklearn.metrics import classification_report, accuracy_score


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, SimpleRNN, LSTM, Flatten, Reshape, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

debug = True

# Load the dataset
train_data = pd.read_csv('finaldata.csv')
test_data = pd.read_csv('finaltest.csv')

# Preprocessing - Scaling features to be between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, 1:32])
X_test = scaler.transform(test_data.iloc[:, 1:32])
y_train = train_data['Target']
y_test = test_data['Target']

# reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the generator
n_inputs = 31
latent_dim = 100

generator = Sequential()
generator.add(Input(shape=(latent_dim,)))
generator.add(Dense(n_inputs, activation='tanh'))
generator.add(Reshape((n_inputs, 1)))

# Define the discriminator
discriminator = Sequential()
discriminator.add(Input(shape=(n_inputs, 1)))
discriminator.add(LSTM(1024, return_sequences=True))
discriminator.add(Dropout(0.4))
discriminator.add(LSTM(512, return_sequences=True))
discriminator.add(Dropout(0.4))
discriminator.add(LSTM(256, return_sequences=True))
discriminator.add(Dropout(0.4))
discriminator.add(Flatten())
discriminator.add(Dense(512))
discriminator.add(LeakyReLU())
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the discriminator
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# The discriminator should be frozen during generator training
discriminator.trainable = False

# Define the combined model
gan_input = Input(shape=(latent_dim,))
fake_sample = generator(gan_input)
gan_output = discriminator(fake_sample)
gan = Model(gan_input, gan_output)

# Compile the GAN
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Define the training procedure
def train(epochs, batch_size=128, sample_interval=50):
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Training the discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_samples = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_samples, valid)
        d_loss_fake = discriminator.train_on_batch(gen_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Training the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Print the progress
        if epoch % sample_interval == 0:
            if debug:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

# Training the GAN
train(epochs=100)  # you can adjust the epochs according to your requirements

# Evaluation of discriminator
y_pred = discriminator.predict(X_test).reshape(-1)
y_pred = [0 if p > 0.5 else 1 for p in y_pred]  # predicting class labels
accuracy = accuracy_score(y_test, y_pred)
if debug:
    print(f"Discriminator accuracy: {accuracy*100:.2f}%")

# Print the classification report
if debug:
    print(classification_report(y_test, y_pred))

# test a sample
# st="0.122,0.303,0.181,0.097,0.219,0.122,0.135,0.284,0.149,0.111,0.46,0.349,0.088,0.509,0.421,0.114,0.617,0.503,0.099,0.21,0.111,0.139,0.225,0.086,0.066,0.371,0.305,0.064,0.492,0.428,0.105"
#Genuine
# st="0.122,1.931,1.809,0.193,0.253,0.06,0.128,0.377,0.249,0.186,0.374,0.188,0.126,0.315,0.189,0.12,0.312,0.192,0.062,0.694,0.632,0.11,0.3,0.19,0.122,0.436,0.314,0.064,0.441,0.377,0.122"
#Imposter
st="0.1,0.517,0.417,0.087,0.446,0.359,0.086,0.255,0.169,0.129,0.422,0.293,0.096,0.663,0.567,0.129,0.61,0.481,0.074,0.181,0.107,0.139,0.214,0.075,0.085,0.39,0.305,0.07,0.366,0.296,0.091"
#Genuine
sample = np.array(st.split(','), dtype=np.float32).reshape(1, -1, 1)
pred = discriminator.predict(sample)
if debug:
    if pred[0][0] > 0.5:
        print('Genuine')
    else:
        print('Imposter')


# In[3]:


import numpy as np
import pandas as pd
from scipy.spatial import distance  # Move this import to the top
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

# Apply MinMaxScaler to the features
scaler = MinMaxScaler()
X_train1 = scaler.fit_transform(data1.iloc[:, 1:32])
X_train2 = scaler.fit_transform(data2.iloc[:, 1:32])
X_test1 = scaler.transform(data1.iloc[:, 1:32])
X_test2 = scaler.transform(data2.iloc[:, 1:32])

# Define labels based on the 'Target' column
y_train = data1['Target'].values
y_test = data2['Target'].values

# Reshape for LSTM
X_train1 = X_train1.reshape((X_train1.shape[0], X_train1.shape[1], 1))
X_train2 = X_train2.reshape((X_train2.shape[0], X_train2.shape[1], 1))
X_test1 = X_test1.reshape((X_test1.shape[0], X_test1.shape[1], 1))
X_test2 = X_test2.reshape((X_test2.shape[0], X_test2.shape[1], 1))

# Define LSTM model
def create_lstm_model():
    input = Input(shape=(X_train1.shape[1], 1))
    x = LSTM(32, return_sequences=True)(input)
    x = LSTM(16)(x)
    return Model(inputs=input, outputs=x)

# Create a single LSTM model for the siamese network
lstm = create_lstm_model()

# Define tensors for two inputs
input1 = Input(shape=(X_train1.shape[1], 1))
input2 = Input(shape=(X_train2.shape[1], 1))

# Process the two inputs
output1 = lstm(input1)
output2 = lstm(input2)

# Compute the absolute difference between the two outputs
diff = Subtract()([output1, output2])

# Add some dense layers and a final binary classifier
diff = Dense(16, activation='relu')(diff)
output = Dense(1, activation='sigmoid')(diff)

# Define the model
model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Fit the model
model.fit([X_train1, X_train2], y_train, validation_data=([X_test1, X_test2], y_test), epochs=10, batch_size=32)

def calculate_distances(arr1, arr2):
    euclidean_dists = []
    manhattan_dists = []

    # Flatten the 2D arrays into 1D for the distance calculations
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()

    euclidean_dists.append(distance.euclidean(arr1_flat, arr2_flat))
    manhattan_dists.append(distance.cityblock(arr1_flat, arr2_flat))

    return euclidean_dists, manhattan_dists

# Compute distances for the training data
euclidean_dists, manhattan_dists = calculate_distances(X_train1, X_train2)

# Compute distances for the test data
euclidean_dists_test, manhattan_dists_test = calculate_distances(X_test1, X_test2)

# Now euclidean_dists and manhattan_dists are lists of distances for each corresponding pair of rows.
# You can print them or further process them as you need.
print(f"Training Data:")
print(f"Euclidean distance: {euclidean_dists[0]}")
print(f"Manhattan distance: {manhattan_dists[0]}")

print(f"Test Data:")
print(f"Euclidean distance: {euclidean_dists_test[0]}")
print(f"Manhattan distance: {manhattan_dists_test[0]}")

# Sequences
seq1 = np.array([0.085, 0.495, 0.41, 0.082, 0.307, 0.225, 0.108, 0.232, 0.124, 0.126, 0.292, 0.166, 0.099, 0.441, 0.342, 0.091, 0.496, 0.405, 0.099, 0.208, 0.109, 0.131, 0.17, 0.039, 0.097, 0.31, 0.213, 0.084, 0.353, 0.269, 0.089])
seq2 = np.array([0.09, 0.376, 0.286, 0.105, 0.215, 0.11, 0.111, 0.239, 0.128, 0.115, 0.37, 0.255, 0.102, 0.574, 0.472, 0.131, 0.61, 0.479, 0.102, 0.241, 0.139, 0.119, 0.23, 0.111, 0.064, 0.28, 0.216, 0.094, 0.381, 0.287, 0.1])

# Calculate Euclidean and Manhattan distances
euclidean_dist = distance.euclidean(seq1, seq2)
manhattan_dist = distance.cityblock(seq1, seq2)

# Print distances
print(f"Euclidean distance: {euclidean_dist}")
print(f"Manhattan distance: {manhattan_dist}")

# Define a threshold (this would typically be determined based on your specific use case)
threshold = 0.5

# Determine whether the sequences are likely from the same user
if euclidean_dist < threshold:
    print("The sequences are likely from the same user.")
else:
    print("The sequences are likely from different users.")


# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam

# Load the dataset
train_data = pd.read_csv('finaldata.csv')
test_data = pd.read_csv('finaltest.csv')

# Preprocessing - Scaling features to be between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, 1:32])
X_test = scaler.transform(test_data.iloc[:, 1:32])
y_train = train_data['Target']
y_test = test_data['Target']

# Define the generator
n_inputs = 31
latent_dim = 100

generator_input = Input(shape=(latent_dim,))
x = Dense(128)(generator_input)
x = LeakyReLU()(x)
x = Dense(n_inputs, activation='tanh')(x)
generator = Model(generator_input, x)

# Define the Discriminator
discriminator_input = Input(shape=(n_inputs,))
x = Dense(1024)(discriminator_input)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.5)(x)
x = Dense(512)(x)
x = LeakyReLU(alpha=0.2)(x)
x = Dropout(0.5)(x)
# ... Continue your architecture as given ...

x = Dense(1, activation='sigmoid')(x)
discriminator = Model(discriminator_input, x)

# Compile the discriminator
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Define the training procedure
def train(epochs, batch_size=128, sample_interval=50):
    # Adversarial ground truths with label smoothing
    valid = np.full((batch_size, 1), 0.9) # real samples labels
    fake = np.zeros((batch_size, 1)) # fake samples labels

    for epoch in range(epochs):
        # Training the discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_samples = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_samples, valid)
        d_loss_fake = discriminator.train_on_batch(gen_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Training the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # Print the progress
        if epoch % sample_interval == 0:
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

# The discriminator should be frozen during generator training
discriminator.trainable = False

# Define the combined model
gan_input = Input(shape=(latent_dim,))
fake_sample_nn = generator(gan_input)
gan_output = discriminator(fake_sample_nn)
gan = Model(gan_input, gan_output)

# Compile the GAN
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Training the GAN
train(epochs=100)

# Check the accuracy of the discriminator for Artificial Neural Network (for the Real data (from finaltest.csv))
y_pred = discriminator.predict(X_test).reshape(-1)
y_pred = [1 if p > 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred)

# Print Discriminator(original data vs original data) accuracy
print(f"Discriminator(Evaluation metrics against original data vs original data) accuracy: {accuracy*100:.2f}%")

# Print the classification report of Discriminator(Evaluation metrics against original data vs original data)
print("Evaluation metrics against original data vs original data is :\n",classification_report(y_test, y_pred))

# Predicting the given string
st = "0.374,3.099,2.725,0.187,0.498,0.311,0.253,0.51,0.257,0.24,2.016,1.776,0.247,0.498,0.251,0.188,0.503,0.315,0.31,3.495,3.185,0.187,0.692,0.505,0.12,0.688,0.568,0.253,1.688,1.435,0.25"
sample = np.array(st.split(','), dtype=np.float32).reshape(1, -1)
pred = discriminator.predict(sample)
if pred[0][0] > 0.5:
    print('Genuine')
else:
    print('Imposter')


# In[5]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from keras.layers import Input, Dense, LeakyReLU, Dropout
from keras.models import Model
from keras.optimizers import Adam

# Load the dataset
train_data = pd.read_csv('finaldata.csv')
test_data = pd.read_csv('finaltest.csv')

# Preprocessing - Scaling features to be between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, 1:32])
X_test = scaler.transform(test_data.iloc[:, 1:32])
y_train = train_data['Target'].values
y_test = test_data['Target'].values

from keras.callbacks import EarlyStopping

# Define Discriminator model
def define_discriminator(n_inputs=31):
    discriminator_input = Input(shape=(n_inputs,))
    x = Dense(2048)(discriminator_input) # Increased units
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x) # Reduced dropout rate
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(discriminator_input, x)
    discriminator.compile(optimizer=Adam(0.00005, 0.5), loss='binary_crossentropy', metrics=['accuracy']) # Reduced learning rate
    return discriminator

# Create and compile the Discriminator
discriminator = define_discriminator()

# Define the training procedure with early stopping
def train(epochs, batch_size=64): # Reduced batch size
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Early Stopping
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    for epoch in range(epochs):
        # Training the discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        fake_samples = np.random.rand(batch_size, X_train.shape[1])
        d_loss_real = discriminator.train_on_batch(real_samples, valid)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Print the progress
        print("%d [D loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1]))

# Training the Discriminator
train(epochs=1000) # Increased epochs

# Rest of the code (testing and evaluation) remains the same


# # Define Discriminator model
# def define_discriminator(n_inputs=31):
#     discriminator_input = Input(shape=(n_inputs,))
#     x = Dense(1024)(discriminator_input)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dropout(0.5)(x)
#     x = Dense(512)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dropout(0.5)(x)
#     x = Dense(256)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dropout(0.5)(x)
#     x = Dense(128)(x)
#     x = LeakyReLU(alpha=0.2)(x)
#     x = Dropout(0.5)(x)
#     x = Dense(1, activation='sigmoid')(x)
#     discriminator = Model(discriminator_input, x)
#     discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
#     return discriminator

# Create and compile the Discriminator
discriminator = define_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Define the training procedure
def train(epochs, batch_size=128):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Training the discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train[idx]
        fake_samples = np.random.rand(batch_size, X_train.shape[1])
        d_loss_real = discriminator.train_on_batch(real_samples, valid)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Print the progress
        print("%d [D loss: %f, acc.: %.2f%%]" % (epoch, d_loss[0], 100*d_loss[1]))

# Training the Discriminator
train(epochs=100)

# Check the accuracy of the Discriminator
y_pred = discriminator.predict(X_test).reshape(-1)
y_pred = [1 if p > 0.5 else 0 for p in y_pred]
accuracy = accuracy_score(y_test, y_pred)

# Print Discriminator(original data vs original data) accuracy
print(f"Discriminator (Evaluation metrics against original data vs original data) accuracy: {accuracy*100:.2f}%")


# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Number of folds
n_splits = 5

# Initialize k-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True)

# Lists to store accuracies
ensemble_accuracies = []
logistic_accuracies = []
# Generating fake samples
noise = np.random.normal(0, 1, (len(X_test[:,1:32]), latent_dim))
fake_samples = generator.predict(noise)

# # Split data into k consecutive folds
# for train_index, test_index in kf.split(X_test_scaled):
#     X_train, X_test = X_test_scaled[train_index], X_test_scaled[test_index]
#     y_train, y_test = y_fake_labels[train_index], y_fake_labels[test_index]

# Ensemble model
ensemble_clf = RandomForestClassifier() # You can tweak hyperparameters here
ensemble_clf.fit(X_train, y_train)
ensemble_preds = ensemble_clf.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
ensemble_accuracies.append(ensemble_accuracy)

# Logistic Regression model
logistic_clf = LogisticRegression() # You can tweak hyperparameters here
logistic_clf.fit(X_train, y_train)
logistic_preds = logistic_clf.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_preds)
logistic_accuracies.append(logistic_accuracy)

# Evaluating the discriminator on fake samples
y_fake_pred =logistic_clf .predict(fake_samples).reshape(-1)
fake_preds_nn = [1 if pred > 0.5 else 0 for pred in y_fake_pred]
y_fake_labels = np.zeros_like(fake_preds_nn ) # Since we want the generator to fool the discriminator into thinking these are real
fake_accuracy = accuracy_score(y_fake_labels, fake_preds_nn )

print(f"Model accuracy on fake data generated by the generator: {fake_accuracy*100:.2f}%")


print(f"Ensemble classifier mean accuracy across {n_splits}-folds: {np.mean(ensemble_accuracies)*100:.2f}%")
print(f"Logistic regression mean accuracy across {n_splits}-folds: {np.mean(logistic_accuracies)*100:.2f}%")


# In[7]:


manhattan_threshold=.5
euclidean_threshold=.5

# Compute distances
manhattan_distances = [distance.cityblock(fake, real) for fake, real in zip(fake_samples, X_test)]
euclidean_distances = [distance.euclidean(fake, real) for fake, real in zip(fake_samples, X_test)]

# Choose appropriate thresholds based on your task and data
manhattan_threshold = np.mean(manhattan_distances) # Example: using the mean distance as a threshold
euclidean_threshold = np.mean(euclidean_distances) # Example: using the mean distance as a threshold

# Create predictions based on distances
manhattan_preds = [1 if d > manhattan_threshold else 0 for d in manhattan_distances]
euclidean_preds = [1 if d > euclidean_threshold else 0 for d in euclidean_distances]

# Compute accuracy
manhattan_accuracy = accuracy_score(y_fake_labels, manhattan_preds)
euclidean_accuracy = accuracy_score(y_fake_labels, euclidean_preds)

print(f"Model accuracy on fake data (Manhattan distance): {manhattan_accuracy*100:.2f}%")
print(f"Model accuracy on fake data (Euclidean distance): {euclidean_accuracy*100:.2f}%")


# In[8]:


from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import numpy as np

# Generating fake samples
noise = np.random.normal(0, 1, (len(X_test), latent_dim))
gen_samples_nn = generator.predict(noise)

# Compute distances
manhattan_distances = [distance.cityblock(fake, real) for fake, real in zip(fake_samples, X_test)]
euclidean_distances = [distance.euclidean(fake, real) for fake, real in zip(fake_samples, X_test)]

# Choose appropriate thresholds based on your task and data
manhattan_threshold = np.mean(manhattan_distances) # Example: using the mean distance as a threshold
euclidean_threshold = np.mean(euclidean_distances) # Example: using the mean distance as a threshold

# Create predictions based on distances
manhattan_preds = [1 if d > manhattan_threshold else 0 for d in manhattan_distances]
euclidean_preds = [1 if d > euclidean_threshold else 0 for d in euclidean_distances]

# Since we want the generator to fool the discriminator into thinking these are real
y_fake_labels = np.zeros_like(manhattan_preds)

# Compute accuracy
manhattan_accuracy = accuracy_score(y_fake_labels, manhattan_preds)
euclidean_accuracy = accuracy_score(y_fake_labels, euclidean_preds)

print(f"Model accuracy on fake data generated by the generator (Manhattan distance): {manhattan_accuracy*100:.2f}%")
print(f"Model accuracy on fake data generated by the generator (Euclidean distance): {euclidean_accuracy*100:.2f}%")


# In[9]:


# from scipy.spatial import distance
# from sklearn.metrics import accuracy_score
# import numpy as np

# # Mock dataset dimensions
# latent_dim = 100
# n_samples = 1000
# n_features = 30

# # Fake generator function
# def generator(noise):
#     return noise * 2

# # Generating fake samples
# X_test = np.random.normal(0, 1, (n_samples, n_features))
# noise = np.random.normal(0, 1, (n_samples, latent_dim))
# fake_samples = generator(noise)

# Compute distances
# manhattan_distances = [distance.cityblock(fake, real) for fake, real in zip(fake_samples[:, :n_features], X_test)]
manhattan_distances = [distance.cityblock(fake, real) for fake, real in zip(fake_samples[:,1:32], X_test)]

euclidean_distances = [distance.euclidean(fake, real) for fake, real in zip(fake_samples[:,1:32], X_test)]

# Choose appropriate thresholds based on your task and data
manhattan_threshold = np.mean(manhattan_distances)
euclidean_threshold = np.mean(euclidean_distances)

# Create predictions based on distances
manhattan_preds = [1 if d > manhattan_threshold else 0 for d in manhattan_distances]
euclidean_preds = [1 if d > euclidean_threshold else 0 for d in euclidean_distances]

# Since we want the generator to fool the discriminator into thinking these are real
y_fake_labels = np.zeros_like(manhattan_preds)

# Compute accuracy
manhattan_accuracy = accuracy_score(y_fake_labels, manhattan_preds)
euclidean_accuracy = accuracy_score(y_fake_labels, euclidean_preds)

print(f"Model accuracy on fake data generated by the generator (Manhattan distance): {manhattan_accuracy*100:.2f}%")
print(f"Model accuracy on fake data generated by the generator (Euclidean distance): {euclidean_accuracy*100:.2f}%")


# In[ ]:


from sklearn.model_selection import KFold
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
import numpy as np

# Number of folds
n_splits = 5

# Initialize k-fold cross-validation
kf = KFold(n_splits=n_splits, shuffle=True)

# Lists to store accuracies
manhattan_accuracies = []
euclidean_accuracies = []

# Split data into k consecutive folds
for train_index, test_index in kf.split(X_test):
    X_train, X_test = X_train[:,1:32], X_test[:,1:32]
    
    # Generate fake samples for the training set
    noise = np.random.normal(0, 1, (len(X_train), latent_dim))
    gen_samples_nn = generator.predict(noise)

    # Compute distances
    manhattan_distances = [distance.cityblock(fake, real) for fake, real in zip(fake_samples[:,1:32], X_test)]
    euclidean_distances = [distance.euclidean(fake, real) for fake, real in zip(fake_samples[:,1:32], X_test)]

    # Choose appropriate thresholds based on your task and data
    manhattan_threshold = np.mean(manhattan_distances)
    euclidean_threshold = np.mean(euclidean_distances)

    # Create predictions based on distances
    manhattan_preds = [1 if d > manhattan_threshold else 0 for d in manhattan_distances]
    euclidean_preds = [1 if d > euclidean_threshold else 0 for d in euclidean_distances]

    # Since we want the generator to fool the discriminator into thinking these are real
    y_fake_labels = np.zeros_like(manhattan_preds)
    y_fake_labels = np.zeros_like( euclidean_preds)


    # Compute accuracy
    manhattan_accuracy = accuracy_score(y_fake_labels, manhattan_preds)
    euclidean_accuracy = accuracy_score(y_fake_labels, euclidean_preds)

    manhattan_accuracies.append(manhattan_accuracy)
    euclidean_accuracies.append(euclidean_accuracy)

print(f"Manhattan distance mean accuracy across {n_splits}-folds: {np.mean(manhattan_accuracies)*100:.2f}%")
print(f"Euclidean distance mean accuracy across {n_splits}-folds: {np.mean(euclidean_accuracies)*100:.2f}%")


# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
train_data = pd.read_csv('trainpattern.csv')
test_data = pd.read_csv('testpattern.csv')

# Preprocessing - Scaling features to be between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, 1:1487])
X_test = scaler.transform(test_data.iloc[:, 1:1487])
y_train = train_data['Target']
y_test = test_data['Target']
# Define the generator
def make_generator():
    noise_input = Input(shape=(100,))
    label_input = Input(shape=(1,))
    x = Concatenate()([noise_input, label_input])
    x = Dense(10, activation='relu')(x)
    x = Dense(1486, activation='sigmoid')(x) # Assuming 1488 numeric attributes
    return Model([noise_input, label_input], x)

# Define the discriminator
def make_discriminator():
    data_input = Input(shape=(1486,))
    label_input = Input(shape=(1,))
    x = Concatenate()([data_input, label_input])
    x = Dense(128, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Dense(256, activation='relu')(x) # Increased to 256 units
    x = Dense(512, activation='relu')(x) # Added another Dense layer
    x = Dense(512, activation='relu')(x)  # Added another Dense layer
    x = Dense(1024, activation='relu')(x)  # Added another Dense layer
    x = Dense(1024, activation='relu')(x)  # Added another Dense layer
    x = Dense(1024, activation='relu')(x)  # Added another Dense layer
    x = Dense(512, activation='relu')(x)   # Added another Dense layer
    x = Dense(256, activation='relu')(x)   # Added another Dense layer
    x = Dense(128, activation='relu')(x)   # Added another Dense layer
    x = Dense(128, activation='relu')(x)    # Added another Dense layer
    x = Dense(128, activation='relu')(x)    # Added another Dense layer
    x = Dense(512, activation='relu')(x)    # Added another Dense layer
    x = Dense(512, activation='relu')(x)     # Added another Dense layer
    x = Dense(1, activation='sigmoid')(x)
    return Model([data_input, label_input], x)

    x = Dense(1, activation='sigmoid')(x)
    return Model([data_input, label_input], x)

    return Model([data_input, label_input], x)

# Compile the models
generator = make_generator()
discriminator = make_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the combined model (for training the generator)
discriminator.trainable = False
noise_input = Input(shape=(100,))
label_input = Input(shape=(1,))
generated_data = generator([noise_input, label_input])
validity = discriminator([generated_data, label_input])
combined = Model([noise_input, label_input], validity)
combined.compile(optimizer='adam', loss='binary_crossentropy')

# Assuming X_train and y_train are NumPy arrays
batch_size = 64
for epoch in range(100):
    # Train discriminator
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_labels = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
    fake_data = generator.predict([noise, fake_labels])
    
    
    idx = np.random.randint(0, X_train.shape[0], batch_size) # Get random indices
    real_data = X_train[idx].astype(np.float32) # Use NumPy array indexing
    real_labels = y_train.values[idx].reshape(-1, 1).astype(np.float32) # Convert to NumPy array, then reshape


    d_loss_real = discriminator.train_on_batch([real_data, real_labels], np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch([fake_data, fake_labels], np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    sampled_labels = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
    g_loss = combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))
    
    
    from sklearn.metrics import accuracy_score
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler



# Reshape y_test to match the shape expected by the model
y_test_reshaped = y_test.values.reshape(-1, 1)

# Use the discriminator to predict the labels for X_test
y_pred = discriminator.predict([X_test, y_test_reshaped]).flatten()

# Convert predictions to binary labels
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Extract the sample and corresponding label
sample_to_predict = X_test[5].reshape(1, -1)
corresponding_label = y_test.values[5].reshape(1, 1)

# Use the discriminator to make a prediction for the sample
predicted_value = discriminator.predict([sample_to_predict, corresponding_label]).flatten()[0]

# Determine whether the prediction indicates "genuine" or "imposter"
if predicted_value > 0.5:
    print("The prediction for X_test[5] is: Genuine")
else:
    print("The prediction for X_test[5] is: Imposter")


    



# In[ ]:


import numpy as np
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load your data
train_data = pd.read_csv('trainpattern.csv')
test_data = pd.read_csv('testpattern.csv')

# Preprocessing - Scaling features to be between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_data.iloc[:, 1:1487]) # Corrected index to include all 1488 attributes
X_test = scaler.transform(test_data.iloc[:, 1:1487])
y_train = train_data['Target']
y_test = test_data['Target']

# Define the generator
def make_generator():
    noise_input = Input(shape=(100,))
    label_input = Input(shape=(1,))
    x = Concatenate()([noise_input, label_input])
    x = Dense(1, activation='relu')(x)
    x = Dense(1486, activation='sigmoid')(x) # Corrected to match 1487 numeric attributes
    return Model([noise_input, label_input], x)

# Define the discriminator
def make_discriminator():
    data_input = Input(shape=(1486,))
    label_input = Input(shape=(1,))
    x = Concatenate()([data_input, label_input])
#     x = Dense(128, activation='relu')(x)
    x = Dense(486, activation='relu')(x)
    x = Dense(486, activation='relu')(x)
    x = Dense(486, activation='relu')(x)
    x = Dense(486, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model([data_input, label_input], x)

# Compile the models
generator = make_generator()
discriminator = make_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the combined model (for training the generator)
discriminator.trainable = False
noise_input = Input(shape=(100,))
label_input = Input(shape=(1,))
generated_data = generator([noise_input, label_input])
validity = discriminator([generated_data, label_input])
combined = Model([noise_input, label_input], validity)
combined.compile(optimizer='adam', loss='binary_crossentropy')

# Assuming X_train and y_train are NumPy arrays
batch_size = 64
for epoch in range(100):
    # Train discriminator
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_labels = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
    fake_data = generator.predict([noise, fake_labels])

    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_data = X_train[idx].astype(np.float32)
    real_labels = y_train.values[idx].reshape(-1, 1).astype(np.float32)

    d_loss_real = discriminator.train_on_batch([real_data, real_labels], np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch([fake_data, fake_labels], np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    sampled_labels = np.random.randint(0, 2, (batch_size, 1)).astype(np.float32)
    g_loss = combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

# Reshape y_test to match the shape expected by the model
y_test_reshaped = y_test.values.reshape(-1, 1)

# Use the discriminator to predict the labels for X_test
y_pred = discriminator.predict([X_test, y_test_reshaped]).flatten()

# Convert predictions to binary labels
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred]

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Extract the sample and corresponding label
sample_to_predict = X_test[7].reshape(1, -1) # Corrected to index 7
corresponding_label = y_test.values[7].reshape(1, 1)

# Use the discriminator to make a prediction for the sample
predicted_value = discriminator.predict([sample_to_predict, corresponding_label]).flatten()[0]

# Determine whether the prediction indicates "genuine" or "imposter"
if predicted_value > 0.5:
    print("The prediction for X_test[7] is: Genuine")
else:
    print("The prediction for X_test[7] is: Imposter")

def check_pattern(data: list[float]) -> str:
    test_data = pd.DataFrame([data], columns=train_data.columns)
    test_data = scaler.transform(test_data.iloc[:, 1:1487])
    sample = test_data[0].reshape(1, -1)
    predicted_val = discriminator.predict([sample, np.array([0])]).flatten()[0]
    return "Genuine" if predicted_val > 0.5 else "Imposter"
# In[ ]:




