const tf = require("@tensorflow/tfjs-node-gpu");
const { resolve } = require("path");
const { existsSync, writeFileSync } = require("fs");
const { ChartJSNodeCanvas } = require("chartjs-node-canvas");
const ChromeLauncher = require("chrome-launcher");

const MODEL_FILE = resolve(__dirname, "models", "number-equality");

const createModel = async () => {
  const inputs = tf.input({
    shape: [2],
    dtype: "int32",
    sparse: false,
  });

  const scaling = tf.layers.rescaling({ scale: 1 / 1000000 }).apply(inputs);

  const hidden = tf.layers
    .dense({ units: 20, activation: "tanh" }) // Looping activator
    .apply(scaling);

  const hidden2 = tf.layers
    .dense({ units: 20, activation: "tanh" }) // Looping activator
    .apply(hidden);

  // const hidden2 = tf.layers
  //   .dense({ units: 20, activation: "elu" }) // Looping activator
  //   .apply(hidden);

  // Regularization
  // const dropout = tf.layers.dropout({ rate: 0.1 }).apply(hidden);

  const outputs = tf.layers
    .dense({ units: 1, activation: "sigmoid" }) // Binary activator
    .apply(hidden2);

  const model = tf.model({ inputs: inputs, outputs });
  compileModel(model);

  return model;
};

const compileModel = (notCompiledModel) => {
  notCompiledModel.compile({
    loss: "meanSquaredError", // Used for optimization
    optimizer: tf.train.adam(),
    // Adjusts learning rate (step size) automatically. SGD fails - weights become NaN due to incorrect learning rate.
    metrics: [tf.metrics.binaryAccuracy], // Just for evaluation of model
  });
};

const loadModel = async () => {
  const model = await tf.loadLayersModel(
    tf.io.fileSystem(resolve(MODEL_FILE, "model.json"))
  );

  compileModel(model);

  return model;
};

const getModel = () => {
  if (existsSync(resolve(MODEL_FILE, "model.json"))) {
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
  let candidate = getRandomNumber(0, 1000000);

  if (candidate == mainNumber) {
    return getBad();
  }

  return candidate;
};

const generateTrainingItems = () => {
  const number = getRandomNumber(0, 1000000);

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
  tf.data.array(generateTrainingData(amountOfPairs)).shuffle(100).batch(100);

const trainingData = generateTrainingDataset(1000);
const validationData = generateTrainingDataset(200);

const earlyStop = tf.callbacks.earlyStopping({
  patience: 50,
  mode: "min",
  verbose: 1,
});

getModel().then(async (model) => {
  const save = {
    // cachedModel: null,
    bestAccuracy: null,

    setParams() {},
    setModel(model) {
      // this.cachedModel = model;
    },
    onTrainBegin() {},
    onEpochBegin() {},
    onBatchBegin() {},
    onBatchEnd() {},
    async onEpochEnd(_epochNumber, { binaryAccuracy }) {
      let needSave = false;
      let oldBestAccuracy = this.bestAccuracy;

      if (!this.bestAccuracy || this.bestAccuracy < binaryAccuracy) {
        this.bestAccuracy = binaryAccuracy;
        needSave = true;
      }

      if (needSave) {
        await model.save(tf.io.fileSystem(MODEL_FILE)).then(() => {
          console.log(
            `Saved best model. Accuracy increased from ${oldBestAccuracy} to ${binaryAccuracy}`
          );
        });
      }
    },
    onTrainEnd() {},
  };

  await model.fitDataset(trainingData, {
    validationData,
    epochs: 1000,
    callbacks: [earlyStop, save],
    verbose: 1,
  });

  const data = [];

  const k = 10000;

  for (let i = 1; i <= 50; i++) {
    for (let j = 1; j <= 50; j++) {
      const result = model.predict(tf.tensor([[i * k, j * k]]));

      if (Math.round(result.dataSync()[0]) > 0.5) {
        data.push({ x: i * k, y: j * k });
      }
    }

    if (i % 5) {
      console.log(i * 2 + "%");
    }
  }

  const renderer = new ChartJSNodeCanvas({ width: 500, height: 500 });

  const dataWithDataset = {
    datasets: [
      {
        label: "Equal numbers",
        data,
        pointBackgroundColor: "red",
      },
    ],
  };

  const graph = renderer.renderToDataURLSync({
    data: dataWithDataset,
    type: "scatter",
    options: {
      scales: {
        x: {
          type: "linear",
          position: "bottom",
        },
        y: {
          type: "linear",
          position: "left",
        },
      },
    },
  });

  const wrappedGraph = `
        <!DOCTYPE html><html><head><title>Graph</title></head><body>
          <img src="${graph}" style="width: 500px; height: 500px;" />
        </body></html>
      `;

  const graphFile = resolve(__dirname, "graph.html");

  writeFileSync(graphFile, wrappedGraph, "utf-8");

  ChromeLauncher.launch({
    startingUrl: "file://" + resolve(__dirname, "graph.html"),
  });

  // model.predict(tf.tensor([[5, 5]])).print();
  // model.predict(tf.tensor([[5, 6]])).print();
  // model.predict(tf.tensor([[5, 7]])).print();
});
