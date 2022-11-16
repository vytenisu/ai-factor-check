const tf = require("@tensorflow/tfjs-node-gpu");
const { resolve } = require("path");
const { existsSync } = require("fs");

const MODEL_FILE = resolve(__dirname, "model.json");

const createModel = async () => {
  const inputs = tf.input({
    shape: [2],
    dtype: "int32",
    sparse: false,
  });

  const hidden = tf.layers
    .dense({ units: 20, activation: "relu" })
    .apply(inputs);

  // const hidden2 = tf.layers
  //   .dense({ units: 20, activation: "relu" })
  //   .apply(hidden);

  const outputs = tf.layers
    .dense({ units: 1, activation: "softmax" })
    .apply(hidden);

  const model = tf.model({ inputs: inputs, outputs });

  model.compile({
    loss: "meanSquaredError",
    // optimizer: "adagrad",
    optimizer: tf.train.adam(100),
    // metrics: [tf.metrics.binaryAccuracy],
    metrics: ["accuracy"],
  });

  return model;
};

const loadModel = () => tf.loadLayersModel("file://" + MODEL_FILE);

const getModel = () => {
  console.log(`Testing if file exists: ${MODEL_FILE}`);

  if (existsSync(MODEL_FILE)) {
    console.log("Loading existent model");
    return loadModel();
  } else {
    console.log("Creating new model");
    return createModel();
  }
};

const getRandomNumber = (min, max) => {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
};

const getGood = (mainNumber) => mainNumber;

const getBad = (mainNumber) => {
  let candidate = getRandomNumber(1, 1000000 - 1);

  if (candidate == mainNumber) {
    return getBad();
  }

  return candidate;
};

const generateTrainingItems = () => {
  const number = getRandomNumber(3, 1000000 - 1);
  return [
    [number, getGood(number)],
    [number, getBad(number)],
  ];
};

const generateTrainingData = (amountOfPairs) => {
  let data = [];

  for (let i = 0; i < amountOfPairs; i++) {
    const [good, bad] = generateTrainingItems();
    data = [...data, { xs: good, ys: [1] }, { xs: bad, ys: [0] }];
  }

  return data;
};

const generateTrainingDataset = (amountOfPairs) =>
  tf.data.array(generateTrainingData(amountOfPairs)).batch(10);

// const trainingData = generateTrainingData(1000);
// const validationData = generateTrainingData(200);

const trainingData = generateTrainingDataset(1000);
const validationData = generateTrainingDataset(200);

getModel().then((model) => {
  // model.fit(tf.tensor([trainingData[1].xs]), tf.tensor([trainingData[1].ys]), {
  //   epochs: 100,
  //   verbose: 1,
  // });

  model.fitDataset(trainingData, {
    validationData,
    epochs: 1,
    // callbacks: [callback],
    verbose: 1,
  });
  // .then(() => {
  //   model.predict(tf.tensor([[5, 5]])).print();
  //   model.predict(tf.tensor([[5, 6]])).print();
  //   model.predict(tf.tensor([[5, 7]])).print();
  // });

  // model.save("file://", MODEL_FILE).then(() => {
});
