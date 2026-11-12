import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from __pycache__.utils import argmax
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



def quantum_entanglement_layer(x):

    batch_outputs = []
    n_qubits = 5

    for sample in x.numpy():

        values = sample[:n_qubits]
        qc = QuantumCircuit(n_qubits)


        # Encode information
        for i in range(n_qubits):
            angle = float(values[i])
            qc.ry(angle,i)


        # Entanglement (Hadamard + CNOT)
        for i in range(n_qubits):
            qc.h(i)

        for i in range(n_qubits-1):
            qc.cx(i,i+1)


        # final state
        state = Statevector.from_instruction(qc)
        probs = state.probabilities()

        # quantum measurement features
        batch_outputs.append(probs)

    return tf.convert_to_tensor(np.array(batch_outputs),dtype=tf.float32)




# Tensorflow wrapper
class QuantumEntanglementLayer(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()


    @tf.custom_gradient
    def quantum_forward(self, inputs):

        quantum_output = tf.py_function(
            func=quantum_entanglement_layer,
            inp=[inputs], Tout=tf.float32)

        quantum_output.set_shape((None,32))

        def grad(dy):
            dy = tf.ones_like(inputs) * tf.reduce_mean(dy)
            return dy

        return quantum_output, grad


    def call(self, inputs):
        return self.quantum_forward(inputs)




def HQEDM(input_shape, classes):
    
    inputs = Input(shape=input_shape)

    # Deep Neural Network Section
    x = Dense(512, activation="relu")(inputs)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)

    # quantum input reduction
    q_input = Dense(5, activation="tanh")(x)

    # Quantum Entanglement Module
    q_features = QuantumEntanglementLayer()(q_input)
    q_features = Dense(64, activation="relu")(q_features)

    # Hybrid Fusion
    fusion = Concatenate()([x, q_features])
    fusion = Dense(128, activation="relu")(fusion)
    fusion = Dropout(0.3)(fusion)

    # Output Layer
    outputs = Dense(classes, activation="softmax")(fusion)
    model = Model(inputs, outputs, name="HQEDM")

    return model




def train_model(X_train, y_train, X_test, y_test):
    
    num_classes = len(np.unique(y_train))
    model = HQEDM(input_shape=(512,), classes=num_classes)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, epochs=300, 
              batch_size=16, validation_data=(X_test, y_test))
    
    return model, history



def predict(X_test):

    model = load_model("Models/Classifier_Model.h5",
        custom_objects={"QuantumEntanglementLayer": QuantumEntanglementLayer})
    
    pred = model.predict(X_test)
    pred = argmax(X_test, axis=1)
    
    return pred








