from keras.models import load_model  # type: ignore
import os
import tensorflow as tf 

def decoderr():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_decoder.h5')
    return load_model(model_path)

def classif_model():
    # model_path=os.path.join(os.path.dirname(__file__), 'models', 'classif_model.h5')
    model_path=os.path.join(os.path.dirname(__file__), 'models', 'best_classifier.h5')

    return load_model(model_path)


# def load_tflite_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.tflite')
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    return interpreter