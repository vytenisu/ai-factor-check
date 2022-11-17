# Random notes

Dependencies:

- Visual Studio 2022 with c++ support
- NVIDIA Cuda
- CuDNN
- WSL
- Python 2.7
- npm install --global @tensorflow/tfjs-node-gpu

Use ml5 for execution in browser

Examples of model creation:

```javascript
// Create an arbitrary graph of layers, by connecting them
// via the apply() method.
const input = tf.input({ shape: [784] }); // shape is only needed for printing things out
// sparse probably means one-hot-encoding - not sure

const dense1 = tf.layers.dense({ units: 32, activation: "relu" }).apply(input);
const dense2 = tf.layers
  .dense({ units: 10, activation: "softmax" })
  .apply(dense1);
const model = tf.model({ inputs: input, outputs: dense2 });
```

```javascript
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
```

Training on one sample at a time:

```javascript
// fit is lower level than fitDataset - it accepts one large tensor instead of Dataset object
model.fit(tf.tensor([trainingData[1].xs]), tf.tensor([trainingData[1].ys]), {
  epochs: 100,
  verbose: 1,
});
```
