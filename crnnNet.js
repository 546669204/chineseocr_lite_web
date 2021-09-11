import BaseModel from "./baseModel.js"
import OcrUtils from "./utils.js";
let keys;

export default class CRNNModel extends BaseModel {
  crnnDstHeight = 32;
  crnnCols = 5531;
  constructor() {
    super("./models/crnn_lite_lstm.onnx");
    this.keyLoading = fetch("./models/keys.txt")
      .then((res) => {
        return res.text();
      })
      .then((res) => {
        keys = res.split("\n");
      });
  }
  async run(partImages) {
    try {
      await this.modelLoading;
      await this.keyLoading;
      return Promise.all(partImages.map(it=>this.GetTextLine(it)));
    } catch (error) {
      console.error(error);
    }
  }
  async GetTextLine(src) {
    let scale = this.crnnDstHeight / src.rows;
    let dstWidth = src.cols * scale;

    let srcResize = new cv.Mat();
    cv.resize(src, srcResize, new cv.Size(dstWidth, this.crnnDstHeight));
    const input = OcrUtils.SubstractMeanNormalize(
      srcResize, [127.5, 127.5, 127.5], [1 / 127.5, 1 / 127.5, 1 / 127.5]
    );

    const feeds = {
      input
    };

    const results = await this.session.run(feeds);
    srcResize.delete();
    return this.ScoreToTextLine(
      results.out.data,
      results.out.size / this.crnnCols,
      this.crnnCols
    );
  }
  ScoreToTextLine(srcData, rows, cols) {
    let lastIndex = 0;
    let scores = [];
    let sb = "";
    for (let i = 0; i < rows; i++) {
      let maxIndex = 0;
      let maxValue = -1000;
      let expList = [];
      for (let j = 0; j < cols; j++) {
        let expSingle = Math.exp(srcData[i * cols + j]);
        expList.push(expSingle);
      }
      let partition = expList.reduce((acc, pre) => acc + pre, 0);
      for (let j = 0; j < cols; j++) {
        let softmax = expList[j] / partition;
        if (softmax > maxValue) {
          maxValue = softmax;
          maxIndex = j;
        }
      }
      if (
        maxIndex > 0 &&
        maxIndex < keys.length &&
        !(i > 0 && maxIndex == lastIndex)
      ) {
        scores.push(maxValue);
        sb += keys[maxIndex - 1];
      }
      lastIndex = maxIndex;
    }
    return sb;
  }
}