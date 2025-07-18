
import numpy
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
# Tokenize the message

f = open("SMSSpamCollection","r")
lines = f.readlines()
labels = []
message = []
for line in lines[:5014]:
  labels.append(line[:4].strip() == "spam")
  message.append(line[4:].strip())
f.close()
labels = numpy.array(labels)
token = []
for i in message:
  token.append(word_tokenize(i))

vocab = set()
for i in token:
  for j in i:
    vocab.add(j)
vocab = list(vocab)
word_to_index = {}
for index,word in enumerate(vocab):
  word_to_index[word] = index + 1 #Reserve first 0 for padding

for i in token: #longest = 69
  for j in range(len(i)):
    i[j] = word_to_index[i[j]]

embedding_dim = 32
model = Sequential([
    Embedding(input_dim=len(vocab) + 1,  # Include 0 for padding
              output_dim=embedding_dim,
              input_length=220,
              mask_zero=True),  # Enable masking
    LSTM(16),
    Dense(1, activation='sigmoid')
])

padded_messages = pad_sequences(token, maxlen=220, padding='post', value=0)

output = model.predict(padded_messages)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    padded_messages,  # Input data
    labels,           # Target labels
    epochs=10,         # Number of epochs
    batch_size=16,    # Batch size
    validation_split=0.2  # Use 20% of the data for validation
)
loss, accuracy = model.evaluate(padded_messages, labels)

#-------------Testing--------------
test_labels = []
test_message = []
for line in lines[5015:5021]:
  test_labels.append(line[:4].strip() == "spam")
  test_message.append(line[4:].strip())
test_labels = numpy.array(test_labels)
tokenized_test = []
for i in test_message:
  tokenized_test.append(word_tokenize(i))
for i in tokenized_test: #longest = 69
  for j in range(len(i)):
    i[j] = word_to_index.get(i[j],0)

test_padded_messages = pad_sequences(tokenized_test, maxlen=69, padding='post', value=0)
test_predictions = model.predict(test_padded_messages)
for i in range(5):  # Show first 5 test samples
    print(f"Message: {test_message[i]}")
    print(f"Actual Label: {'Spam' if test_labels[i] == 1 else 'Not Spam'}")
    print(f"Predicted Probability: {test_predictions[i][0]:.2f}")
    print(f"Predicted Label: {'Spam' if test_predictions[i][0] > 0.5 else 'Not Spam'}\n")