import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.dual_encoder import dual_encoder_model
from models.helpers import load_vocab
import pandas as pd
from termcolor import colored

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load data for predict
test_df = pd.read_csv("./data/test.csv")
#elementId = 9 #15002
elementId = 27 #15002
#elementId = 32 #15002
#elementId = 33 #15002
INPUT_CONTEXT = test_df.Context[elementId]
POTENTIAL_RESPONSES = test_df.iloc[elementId,1:].values

test_context = ["how can i change my wallpaper?",
      "hi guy, how are you? i heard you went to the Alps",
      "como debo cambiar mi password en ubuntu desde la consola"]

test_reponses = [["Right click on the desktop and chose change wallpaper",
                        "go the options and seek the option",
                        "press the button and reboot your computer.", 
                        "click wallpaper, wallpaper",
                        "Select the Start Start symbol button, then select Settings  > Personalization to choose a picture worthy of gracing your desktop background",
                        "Right-click your desktop and select Set wallpaper. Click one of the images to set your wallpaper. You can also get a randomly selected image by checking the box next to Surprise me"],
          ["you're welcome, your ears are fine",
              "no, my dad was scientist",
              "hi man, i am very happy to have gone, it is a wonderful place",
              "I'm fantastic",
              "I'm a little sad, my girlfriend broke up with me",
              "my dog ate something bad",
              "I'm better, yesterday I went to the circus"],
          ["abre la terminal y escribe $sudo reboot",
                       "escribe en la terminal $sudo apt-get install geany, cuando termine abre geany y cambia tu password",
                        "abre la ventana de configuracion de red y busca redes disponibles marca el check de redes ocultas", 
                        "escribe en la consola $sudo shutdown -now",
                        "entra en la terminal con tu usuario y escribe $sudo passwd, entonces pon tu password actual, luego pon el nuevo",
                        "haz click derecho en el escritorio, ve a la opcion cambiar password, despues sigue las instrucciones"]

]

#my_idx = 1
#INPUT_CONTEXT = test_context[my_idx]
#POTENTIAL_RESPONSES = test_reponses[my_idx]

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  # Ugly hack, seems to be a bug in Tensorflow
  # estimator.predict doesn't work without this line
  estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  probs = []
  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    results = next(prob)
    probs.append(results[0])
    #print("{:g}: {}".format(results[0],r))
    #print("{}: {:g}".format(r, prob[0,0]))

  nprobs = np.array(probs)
  idMax = nprobs.argmax(axis=0)
  print()

  print(colored('[     Context]', on_color='on_blue',color="yellow"),INPUT_CONTEXT)
  
  print()
  print(colored("[   Estimated]", on_color='on_white', color="blue"))
  for i in range(0,len(POTENTIAL_RESPONSES)):
    print("[{:g}]: {}".format(probs[i],POTENTIAL_RESPONSES[i]))
  print()
  print(colored('[     Context]', on_color='on_blue',color="yellow"),INPUT_CONTEXT)
  print(colored('[      Answer]', on_color='on_green', color="white"),POTENTIAL_RESPONSES[idMax])

  print()
