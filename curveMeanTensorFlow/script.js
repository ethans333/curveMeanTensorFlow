let model;

const f = (x) => eval(document.getElementById("txtFunction").value.replace("x", x.toString()));

const Svalues = {
  values: () => parseInt(document.getElementById('nValues').value),
  layers: () => parseInt(document.getElementById('nLayers').value),
  epochs: () => parseInt(document.getElementById('nEpochs').value),
  threshold: () => parseInt(document.getElementById('nThreshold').value)
}

const getData = (n) => [...Array(n).keys()].map(i => {
    return { 
      a: i*100/n, 
      b: f(i*100/n) + Math.random()*Math.max(...[...Array(n).keys()].map(j => f(j*100/n)))/(11-Svalues.threshold())
    }
});

const createModel = (nLayers) => {
    const model = tf.sequential();

    model.add(tf.layers.dense({inputShape: [1], units: 1}));

    for (let i = 0; i < nLayers; i++) {
        model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    }

    model.add(tf.layers.dense({units: 1}));

    return model;
}

const convertToTensor = (data) => tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map(d => d.a), labels = data.map(d => d.b);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]), labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();

    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
        inputs: normInputs,
        labels: normLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin,
    }
});

const trainModel = async (model, inputs, labels) => {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32, epochs = Svalues.epochs();

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

const testModel = (model, inputData, normalizationData) => {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    const [xs, preds] = tf.tidy(() => {

        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));
    
        const unNormXs = xs
          .mul(inputMax.sub(inputMin))
          .add(inputMin);
    
        const unNormPreds = preds
          .mul(labelMax.sub(labelMin))
          .add(labelMin);
    
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
      });

      const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
      });
    
      const originalPoints = inputData.map(d => ({
        x: d.a, y: d.b,
      }));
    
    
      tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
        {
          xLabel: 'x',
          yLabel: document.getElementById("txtFunction").value,
          height: 300
        }
      );
}

const run = async () => {
    const data = getData(Svalues.values());
    console.log(data)

    const values = data.map(d => ({
      x: d.a,
      y: d.b,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Plotted Curve w/ Random Values'},
      {values},
      {
        xLabel: 'x',
        yLabel: document.getElementById("txtFunction").value,
        height: 300
      }
    );
    
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    await trainModel(model, inputs, labels);
    console.log('Done Training');

    testModel(model, data, tensorData);
}

const elIds = ["Values", "Layers", "Epochs"];

elIds.forEach(id => {
  document.getElementById(`n${id}`).onchange = () => {
    document.getElementById(`lbl${id}`).innerHTML = `${id}: ${document.getElementById(`n${id}`).value}`;
  }
})

document.getElementById("nThreshold").onchange = () => {
  document.getElementById("lblThreshold").innerHTML = `Random Offset Threshold: ${Svalues.threshold()}`;
}

document.getElementById("train").onclick = () => {
  document.getElementById("lblFunction").innerHTML = `Function: ${document.getElementById("txtFunction").value}`;

  model = createModel(Svalues.layers());
  tfvis.show.modelSummary({name: 'Model Summary'}, model);
  run();
};