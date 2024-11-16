<<<<<<< HEAD
# ML2
=======
# starter code for a3

Add the corresponding (one) line under the ``[to fill]`` in ``def forward()`` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

Our benmarking performance after completing def forward() code:

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 ``
``--train_data ./training.json --val_data ./validation.json``

validation acc >=0.43

**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 ``
``--train_data training.json --val_data validation.json``

validation acc >= 0.31
>>>>>>> 28b4214 (add init)
