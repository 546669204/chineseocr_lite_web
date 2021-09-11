import BaseModel from "./baseModel.js"
import OcrUtils from "./utils.js";

class Angle {

}
export default class angleNetModel extends BaseModel {
  angleDstWidth = 192;
  angleDstHeight = 32;
  angleCols = 2;
  constructor() {
    super("./models/angle_net.onnx");
  }
  AdjustTargetImg(src, dstWidth, dstHeight) {
    let srcResize = new cv.Mat();
    let scale = dstWidth / src.cols;
    let angleHeight = parseInt(src.rows * scale);
    cv.resize(src, srcResize, new cv.Size(dstWidth, angleHeight));
    let srcFit = new cv.Mat(dstHeight, dstWidth, cv.CV_8U);
    if (angleHeight < dstHeight) {
      cv.copyMakeBorder(srcResize, srcFit, 0, dstHeight - angleHeight, 0, 0, cv.BORDER_ISOLATED, new cv.Scalar(255, 255, 255));
    } else {
      let rect = new cv.Rect(0, 0, dstWidth, dstHeight);
      let partAngle = srcResize.roi(rect);
      partAngle.copyTo(srcFit);
    }
    srcResize.delete();
    return srcFit;
  }
  ScoreToAngle(srcData, angleCols) {
    let angle = new Angle();
    let angleIndex = 0;
    let maxValue = -1000.0;
    for (let i = 0; i < angleCols; i++) {
      if (i == 0) maxValue = srcData[i];
      else if (srcData[i] > maxValue) {
        angleIndex = i;
        maxValue = srcData[i];
      }
    }
    angle.Index = angleIndex;
    angle.Score = maxValue;
    return angle;
  }
  async GetAngle(src) {
    let angleImg = this.AdjustTargetImg(src, this.angleDstWidth, this.angleDstHeight);
    const inputTensors = OcrUtils.SubstractMeanNormalize(
      angleImg, [127.5, 127.5, 127.5], [1 / 127.5, 1 / 127.5, 1 / 127.5]
    );
    const feeds = {
      input: inputTensors
    };
    const results = await this.session.run(feeds);
    return this.ScoreToAngle(results.out.data, this.angleCols);
  }
  async run(partImgs, doAngle, mostAngle) {
    await this.modelLoading;
    let angles = [];
    if (doAngle) {
      for (let i = 0; i < partImgs.length; i++) {
        var angle = await this.GetAngle(partImgs[i]);
        angles.push(angle);
      }
    } else {
      for (let i = 0; i < partImgs.length; i++) {
        var angle = new Angle();
        angle.Index = -1;
        angle.Score = 0;
        angles.push(angle);
      }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
      let angleIndexes = 0;
      angles.forEach(x => angleIndexes += x.Index);

      let sum = angleIndexes;
      let halfPercent = angles.length / 2.0;
      let mostAngleIndex;
      if (sum < halfPercent) { //all angle set to 0
        mostAngleIndex = 0;
      } else { //all angle set to 1
        mostAngleIndex = 1;
      }
      for (let i = 0; i < angles.length; ++i) {
        let angle = angles[i];
        angle.Index = mostAngleIndex;
        angles[i] = angle;
      }
    }
    return angles;
  }
}