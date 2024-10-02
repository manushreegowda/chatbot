!pip install keras 
!pip install nltk
!pip install tensorflow
import nltk
import numpy as np
import random
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers # make sure to import keras this way


nltk.download('punkt')
nltk.download('wordnet')

# Sample data (replace with your own dataset)
data = {
    "greetings": ["hello", "hi", "hey"],
    "goodbyes": ["bye", "goodbye", "see you later"],
    "thanks": ["thank you", "thanks", "thanks a lot"],
    "responses": {
        "greetings": ["Hello there!", "Hi!", "Hey!"],
        "goodbyes": ["Goodbye!", "See you later!", "Take care!"],
        "thanks": ["You're welcome!", "No problem!", "Glad I could help!"]
    }
}

# Tokenization and preprocessing
word_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
lemmatizer = nltk.stem.WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenizer.tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Create vocabulary
vocabulary = set()
for intent, examples in data.items():
    if intent != "responses":
        for example in examples:
            vocabulary.update(preprocess(example))

vocabulary = list(vocabulary)
word_to_index = {word: index for index, word in enumerate(vocabulary)}
index_to_word = {index: word for index, word in enumerate(vocabulary)}

# Prepare training data
training_data = []
for intent, examples in data.items():
    if intent != "responses":
        for example in examples:
            tokens = preprocess(example)
            input_vector = [word_to_index[word] for word in tokens if word in word_to_index]
            output_vector = [intent]
            training_data.append((input_vector, output_vector))

# Define the model, make sure to use layers from tensorflow.keras
model = tf.keras.Sequential([ # Use tf.keras.Sequential
    layers.Embedding(len(vocabulary), 128), # Use layers from tensorflow.keras
    layers.LSTM(128),
    layers.Dense(len(data) - 1, activation='softmax')  # Number of intents
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
X = np.array([np.array(input_vector) for input_vector, _ in training_data])
Y = np.array([list(data.keys()).index(output_vector[0]) for _, output_vector in training_data])

X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, padding='post')

model.fit(X_padded, Y, epochs=500)


# Chat function
def chatbot_response(user_input):
    tokens = preprocess(user_input)
    input_vector = [word_to_index[word] for word in tokens if word in word_to_index]
    input_vector_padded = tf.keras.preprocessing.sequence.pad_sequences([input_vector], padding='post', maxlen=X_padded.shape[1])
    prediction = model.predict(input_vector_padded)[0]
    predicted_intent_index = np.argmax(prediction)
    predicted_intent = list(data.keys())[predicted_intent_index]

    if predicted_intent in data["responses"]:
        response = random.choice(data["responses"][predicted_intent])
        return response
    else:
        return "I'm sorry, I didn't understand that."

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break