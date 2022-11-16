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

  const outputs = tf.layers
    .dense({ units: 1, activation: "softmax" })
    .apply(hidden);

  const model = tf.model({ inputs: inputs, outputs });

  model.compile({
    loss: "meanSquaredError",
    // optimizer: "adagrad",
    optimizer: "adam",
    metrics: [tf.metrics.binaryAccuracy],
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

// const getRandomArrayELement = (data) =>
//   data[Math.floor(Math.random() * data.length)];

const getGoodFactor = (mainNumber) => {
  let candidateFactor = getRandomNumber(1, mainNumber);

  while (mainNumber % candidateFactor) {
    candidateFactor = getRandomNumber(1, mainNumber);
  }

  return candidateFactor;
};

const getBadFactor = (mainNumber) => {
  let candidateFactor = getRandomNumber(1, mainNumber);

  while (!(mainNumber % candidateFactor)) {
    candidateFactor = getRandomNumber(1, mainNumber);
  }

  return candidateFactor;
};

const generateTrainingItems = () => {
  const number = getRandomNumber(3, 1000000 - 1);
  return [
    [number, getGoodFactor(number)],
    [number, getBadFactor(number)],
  ];
};

const generateTrainingData = (amountOfPairs) => {
  let data = [];
  // let labels = [];

  for (let i = 0; i < amountOfPairs; i++) {
    const [good, bad] = generateTrainingItems();
    data = [...data, { xs: good, ys: 1 }, { xs: bad, ys: 0 }];
    // labels.push(1);
    // labels.push(0);
  }

  // const xs = tf.tensor2d(data, [data.length, 2]);
  // const ys = tf.tensor1d(labels);

  // tf.data.csv({json: })

  return tf.data.array(data).batch(10);
};

const trainingData = generateTrainingData(1000);
const validationData = generateTrainingData(200);

// const callback = tf.callbacks.earlyStopping({
//   patience: 3,
//   monitor: "loss",
//   verbose: 1,
// });

console.log(trainingData);
console.log(validationData);

// console.log(generateTrainingData(50));

// new tf.data.array([])

// tf.tensor([12345, 4]).print();
// tf.randomUniform([2], 1, 1000, "int32").print();
// tf.randomUniform([2], 1, 1000, "int32").print();
// tf.randomUniform([2], 1, 1000, "int32").print();

getModel().then((model) => {
  model.fitDataset(trainingData, {
    validationData,
    epochs: 100,
    // callbacks: [callback],
    verbose: 1,
  });
  // model.save("file://", MODEL_FILE).then(() => {
  // console.log(
  //   model.weights.map(({ name, shape }) => ({
  //     name,
  //     shape,
  //   }))
  // );
  // });
});
