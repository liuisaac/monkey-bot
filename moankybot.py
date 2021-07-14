import discord
import random
import os
from discord.ext import commands, tasks

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time



#==========================================================================================================#


intents = discord.Intents(messages = True, guilds = True, reactions = True, members = True, presences = True)
client = commands.Bot(command_prefix = '.', intents = intents)

path_to_file = "C:\\Users\\Isaac Liu\\Downloads\\moankyserver.txt"

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it

vocab = sorted(set(text))

#print(text)

for filename in os.listdir('C:\\Users\\Isaac Liu\\Desktop\\Python Folder\\PyTorch\\cogs'):
  if filename.endswith('.py'):
    client.load_extension(f'cogs.{filename[:-3]}')

#processing and vectorizing
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab))

ids = ids_from_chars(chars)

#for reverting vectorization
chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True)

tf.strings.reduce_join(chars, axis=-1).numpy()

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

#By here, our data is converted into a tf.Tensor with a shape of (9736,) and vecotrize. Next, we must build the model
#char->num

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
all_ids


ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)


seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text
  
dataset = sequences.map(split_input_target)



# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024
print(" ")
#model
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x

model = MyModel(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

model.summary()

print(" ")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

sampled_indices

print(" ")

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
mean_loss = example_batch_loss.numpy().mean()
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", mean_loss)

tf.exp(mean_loss).numpy()

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 60

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "" or "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)


#--------=================--------------------====================------------------==============-----------==========#

@client.command()
async def load(ctx, extension):
  client.load_extension(f'cogs.{extension}')

@client.command()
async def unload(ctx, extension):
  client.unload_extension(f'cogs.{extension}')

@client.event 
async def on_ready():
    await client.change_presence(status=discord.Status.online, activity=discord.Game(':monkey: gr'))
    print("grrr whomst has awoken ME")

@client.command(aliases = ['gen', 'speaky'])
async def speak2(ctx):
    start = time.time()
    states = None
    next_char = tf.constant(['**moanky has a message:** Its'])
    result = [next_char]

    for n in range(100):
      next_char, states = one_step_model.generate_one_step(next_char, states=states)
      result.append(next_char)

    fresult = str(tf.strings.join(result)[0].numpy().decode("utf-8"))
    await ctx.send(f"{fresult}")
    fresult = []

@client.command()
async def sellmesomething(ctx):
    buylist = ["https://www.amazon.ca/Aurora-Monkey-12-Inch-Flopsie-Stuffed/dp/B003231HVI/ref=pd_lpo_21_t_1/134-6182451-4178244?_encoding=UTF8&pd_rd_i=B003231HVI&pd_rd_r=66b976d5-9033-4172-b842-d7e8aae54b5d&pd_rd_w=SfdSt&pd_rd_wg=gblim&pf_rd_p=83e3102e-d62a-49fe-be3a-977069c19060&pf_rd_r=RD0DTDNKPYFX6W2N24W8&psc=1&refRID=RD0DTDNKPYFX6W2N24W8","https://www.amazon.ca/YunNasi-Monkey-Stuffed-Animal-Cuddly/dp/B07PYN3QXC","https://yzgaotai.en.made-in-china.com/product/qXgmuLVHgzYK/China-Cute-Mischief-Monkey-Stuffed-Animal-Plush-Fat-Soft-Toy.html","https://www.alibaba.com/product-detail/Promotional-gifts-fat-monkey-plush-wholesale_60108013415.html","joom.com/en/products/5cb44a1236b54d01014c6235","https://www.hashtagcollectibles.com/products/colossal-proboscis-monkey-plush","https://www.ebay.com/itm/264898627882","https://www.alibaba.com/product-detail/1pcs-6-3-16cm-Wild-Republic_60606320436.html","https://www.walmart.com/ip/Multipet-Swingin-Safari-Plush-Squeaky-Monkey-Dog-Toy/15786680","https://taihuatoy.en.made-in-china.com/product/hyYmaopOHrkH/China-Wholesale-Baby-Soft-Animal-Plush-Brown-Monkey-Stuffed-Toy.html","https://www.alzashop.com/toys/beanie-boos-coconut-monkey-d5023023.htm","https://www.worthpoint.com/worthopedia/sweet-sprouts-animal-adventure-brown-1832829347","http://store.fieldmuseum.org/products/realistic-capuchin-monkey-plush","https://www.getdigital.eu/three-headed-monkey-plush.html","https://www.alzashop.com/toys/super-mario-monkey-d4845453.htm","https://www.dailymail.co.uk/news/article-9405567/Obese-Thai-monkey-weight-DOUBLED-eating-junk-food-sent-fat-camp-shape.html","https://www.aliexpress.com/item/32238341448.html","https://www.amazon.com/Monkey-Stuffed-Animal-Kawaii-Childern/dp/B08NC5FHWW","https://www.aliexpress.com/i/32967014978.html","https://www.ebay.com/itm/264375172537","https://www.aliexpress.com/item/32965141787.html","https://www.alibaba.com/product-detail/Round-fat-like-ball-soft-stuffed_60708372557.html"]
    bmsg = str("YOU just winned the (insert text) !!! HURYY AMd TYPE .bUy NOW!!! huryy "+ str(buylist[random.randint(0,22)]))
    await ctx.send(bmsg)
    time.sleep(3)
    await ctx.send("hruy gareth theres not much time left!!")


client.run('ODE4MjA3MTQ1NjM5ODA0OTM5.YEUsyQ.C3CcoIwhXXE8NXWmpLqzI0Zd75M')