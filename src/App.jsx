import { useCallback, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { loadGraphModel } from "@tensorflow/tfjs-converter";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import Canvas from "./Canvas";

function App() {
  const [bigCanvas, setBigCanvas] = useState(null);
  const [smallCanvas, setSmallCanvas] = useState(null);
  const [models, setModels] = useState({});
  const [prediction, setPrediction] = useState({});

  useEffect(() => {
    (async () => {
      try {
        console.log("Loading models...");
        setModels({
          modelConvAug: await loadGraphModel(
            "./models/mnist-conv-augmentation/model.json"
          ),
          modelConv: await loadGraphModel("./models/mnist-conv/model.json"),
          modelDense: await loadGraphModel("./models/mnist-dense/model.json"),
        });
        console.log("Models loaded...");
      } catch (error) {
        console.log(error.message);
      }
    })();
  }, []);

  const handleCanvasReady = useCallback((canvas) => {
    if (canvas.getWidth() === 200) {
      setBigCanvas(canvas);
    } else {
      setSmallCanvas(canvas);
    }
  }, []);

  const resample_single = (canvas, width, height, resize_canvas) => {
    var width_source = canvas.width;
    var height_source = canvas.height;
    width = Math.round(width);
    height = Math.round(height);

    var ratio_w = width_source / width;
    var ratio_h = height_source / height;
    var ratio_w_half = Math.ceil(ratio_w / 2);
    var ratio_h_half = Math.ceil(ratio_h / 2);

    var ctx = canvas.getContext("2d");
    var ctx2 = resize_canvas.getContext("2d");
    var img = ctx.getImageData(0, 0, width_source, height_source);
    var img2 = ctx2.createImageData(width, height);
    var data = img.data;
    var data2 = img2.data;

    for (var j = 0; j < height; j++) {
      for (var i = 0; i < width; i++) {
        var x2 = (i + j * width) * 4;
        var weight = 0;
        var weights = 0;
        var weights_alpha = 0;
        var gx_r = 0;
        var gx_g = 0;
        var gx_b = 0;
        var gx_a = 0;
        var center_y = (j + 0.5) * ratio_h;
        var yy_start = Math.floor(j * ratio_h);
        var yy_stop = Math.ceil((j + 1) * ratio_h);
        for (var yy = yy_start; yy < yy_stop; yy++) {
          var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half;
          var center_x = (i + 0.5) * ratio_w;
          var w0 = dy * dy; //pre-calc part of w
          var xx_start = Math.floor(i * ratio_w);
          var xx_stop = Math.ceil((i + 1) * ratio_w);
          for (var xx = xx_start; xx < xx_stop; xx++) {
            var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half;
            var w = Math.sqrt(w0 + dx * dx);
            if (w >= 1) {
              //pixel too far
              continue;
            }
            //hermite filter
            weight = 2 * w * w * w - 3 * w * w + 1;
            var pos_x = 4 * (xx + yy * width_source);
            //alpha
            gx_a += weight * data[pos_x + 3];
            weights_alpha += weight;
            //colors
            if (data[pos_x + 3] < 255)
              weight = (weight * data[pos_x + 3]) / 250;
            gx_r += weight * data[pos_x];
            gx_g += weight * data[pos_x + 1];
            gx_b += weight * data[pos_x + 2];
            weights += weight;
          }
        }
        data2[x2] = gx_r / weights;
        data2[x2 + 1] = gx_g / weights;
        data2[x2 + 2] = gx_b / weights;
        data2[x2 + 3] = gx_a / weights_alpha;
      }
    }

    for (var p = 0; p < data2.length; p += 4) {
      var gray = data2[p];

      if (gray < 100) {
        gray = 0;
      } else {
        gray = 255;
      }

      data2[p] = gray;
      data2[p + 1] = gray;
      data2[p + 2] = gray;
    }

    ctx2.putImageData(img2, 0, 0);
  };

  const predict = () => {
    resample_single(bigCanvas, 28, 28, smallCanvas);

    var imgData = smallCanvas.getContext("2d").getImageData(0, 0, 28, 28);
    var arr = [];
    var arr28 = [];
    for (var p = 0; p < imgData.data.length; p += 4) {
      var value = imgData.data[p + 3];
      arr28.push([value]);
      if (arr28.length == 28) {
        arr.push(arr28);
        arr28 = [];
      }
    }

    arr = [arr];
    var tensor4 = tf.tensor4d(arr, [1, 28, 28, 1], "int32");
    setPrediction({
      predictionConvAug: models.modelConvAug.predict(tensor4).dataSync(),
      predictionConv: models.modelConv.predict(tensor4).dataSync(),
      predictionDense: models.modelDense.predict(tensor4).dataSync(),
    });
  };

  return (
    <>
      <div>
        <a href="https://vitejs.dev" target="_blank" rel="noreferrer">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank" rel="noreferrer">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Draw a number</h1>
      <div className="card">
        <div className="canvas-area">
          <Canvas
            id={"bigC"}
            height={200}
            width={200}
            setClear={true}
            onCanvasReady={handleCanvasReady}
            className="canvas"
          />
          <button onClick={predict}>Predict</button>
        </div>
        <div>
          <h2>MODELS</h2>
          <table>
            <thead>
              <tr>
                <th>Model Dense</th>
                <th>Model Conv</th>
                <th>Model Conv + Aug</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  {prediction.predictionDense === undefined
                    ? ""
                    : prediction.predictionDense}
                </td>
                <td>
                  {prediction.predictionConv === undefined
                    ? ""
                    : prediction.predictionConv}
                </td>
                <td>
                  {prediction.predictionConvAug === undefined
                    ? ""
                    : prediction.predictionConvAug}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <Canvas
          id={"smallC"}
          height={28}
          width={28}
          onCanvasReady={handleCanvasReady}
          className="canvas smallC"
        />
      </div>
    </>
  );
}

export default App;
