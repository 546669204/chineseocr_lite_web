import ndarray from "https://cdn.skypack.dev/-/ndarray@v1.0.19-grdQeKOTBdxK6FqCxCHR/dist=es2020,mode=imports/optimized/ndarray.js";
import ops from "https://cdn.skypack.dev/-/ndarray-ops@v1.2.2-beRu4E9rjWhoa5R2nJLo/dist=es2020,mode=imports/optimized/ndarray-ops.js";

export default class OcrUtils {
  static MakePadding(src, padding) {
    if (padding <= 0) return src;
    let paddingScalar = new cv.Scalar(255, 255, 255);
    let paddingSrc = new cv.Mat();
    cv.copyMakeBorder(
      src,
      paddingSrc,
      padding,
      padding,
      padding,
      padding,
      cv.BORDER_ISOLATED,
      paddingScalar
    );
    return paddingSrc;
  }
  static SubstractMeanNormalizeCanvas(ctx, meanVals, normVals) {
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const {
      data,
      width,
      height
    } = imageData;
    // r c chs:
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );
    for (let ch = 0; ch < 3; ch++) {
      ops.mulseq(dataProcessedTensor.pick(0, ch, null, null), normVals[ch]);
      ops.subseq(
        dataProcessedTensor.pick(0, ch, null, null),
        meanVals[ch] * normVals[ch]
      );
    }

    const tensor = new ort.Tensor(
      "float32",
      new Float32Array(width * height * 3), [1, 3, width, height]
    );
    tensor.data.set(dataProcessedTensor.data);

    return tensor;
  }
  static SubstractMeanNormalize(ctx, meanVals, normVals) {
    const {
      data,
      cols: width,
      rows: height
    } = ctx;
    // data processing
    // r c chs:
    const dataTensor = ndarray(data, [height, width, ctx.channels()]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * ctx.channels()), [
      1,
      ctx.channels(),
      height,
      width,
    ]);
    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );
    for (let ch = 0; ch < ctx.channels(); ch++) {
      ops.mulseq(dataProcessedTensor.pick(0, ch, null, null), normVals[ch]);
      ops.subseq(
        dataProcessedTensor.pick(0, ch, null, null),
        meanVals[ch] * normVals[ch]
      );
    }

    const tensor = new ort.Tensor(
      "float32",
      new Float32Array(width * height * ctx.channels()), [1, ctx.channels(), height, width]
    );
    tensor.data.set(dataProcessedTensor.data);

    return tensor;
  }
  static GetThickness(boxImg) {
    let minSize = boxImg.cols > boxImg.rows ? boxImg.rows : boxImg.cols;
    let thickness = minSize / 1000 + 2;
    return thickness;
  }
  static DrawTextBox(boxImg, box, thickness) {
    if (box == null || box.length == 0) {
      return;
    }
    var color = new cv.Scalar(255, 0, 0); //R(255) G(0) B(0)
    cv.line(boxImg, box[0], box[1], color, thickness);
    cv.line(boxImg, box[1], box[2], color, thickness);
    cv.line(boxImg, box[2], box[3], color, thickness);
    cv.line(boxImg, box[3], box[0], color, thickness);
  }
  static DrawTextBoxes(src, textBoxes, thickness) {
    for (let i = 0; i < textBoxes.length; i++) {
      let t = textBoxes[i];
      this.DrawTextBox(src, t.Points, thickness);
    }
  }
  static GetRotateCropImage(src, box) {
    let image = new cv.Mat();
    src.copyTo(image);

    let points = box.slice();

    let collectX = [box[0].x, box[1].x, box[2].x, box[3].x];
    let collectY = [box[0].y, box[1].y, box[2].y, box[3].y];
    let left = Math.min.apply(null, collectX);
    let right = Math.max.apply(null, collectX);
    let top = Math.min.apply(null, collectY);
    let bottom = Math.max.apply(null, collectY);

    let rect = new cv.Rect(left, top, right - left, bottom - top);
    let imgCrop = image.roi(rect);

    for (let i = 0; i < points.length; i++) {
      var pt = new cv.Point(points[i].x,points[i].y);
      pt.x -= left;
      pt.y -= top;
      points[i] = pt;
    }

    let imgCropWidth = Math.sqrt(
      Math.pow(points[0].x - points[1].x, 2) +
      Math.pow(points[0].y - points[1].y, 2)
    );
    let imgCropHeight = Math.sqrt(
      Math.pow(points[0].x - points[3].x, 2) +
      Math.pow(points[0].y - points[3].y, 2)
    );


    let ptsDst = [];
    ptsDst.push(0, 0);
    ptsDst.push(imgCropWidth, 0);
    ptsDst.push(imgCropWidth, imgCropHeight);
    ptsDst.push(0, imgCropHeight);

    let ptsSrc = [];
    ptsSrc.push(points[0].x, points[0].y);
    ptsSrc.push(points[1].x, points[1].y);
    ptsSrc.push(points[2].x, points[2].y);
    ptsSrc.push(points[3].x, points[3].y);

    let M = cv.getPerspectiveTransform(
      cv.matFromArray(4, 1, cv.CV_32FC2, ptsSrc),
      cv.matFromArray(4, 1, cv.CV_32FC2, ptsDst),
    );

    let partImg = new cv.Mat();
    cv.warpPerspective(
      imgCrop,
      partImg,
      M,
      new cv.Size(imgCropWidth, imgCropHeight),
      cv.INTER_NEAREST,
      cv.BORDER_REPLICATE,
      new cv.Scalar(255,255,255)
    );
    try {
      if (partImg.rows >= partImg.cols * 1.5) {
        let srcCopy = new cv.Mat();
        cv.transpose(partImg, srcCopy);
        cv.flip(srcCopy, srcCopy, 0);
        return srcCopy;
      } else {
        return partImg;
      }
    } finally {
      image.delete();
      imgCrop.delete();
    }

  }
  static GetPartImages(src, textBoxes) {
    let partImages = [];
    for (let i = 0; i < textBoxes.length; ++i) {
      let partImg = this.GetRotateCropImage(src, textBoxes[i].Points);
      //Mat partImg = new Mat();
      //GetRoiFromBox(src, partImg, textBoxes[i].Points);
      partImages.push(partImg);
    }
    return partImages;
  }
  static MatRotateClockWise180(src) {
    // cv.flip(src, src, FlipType.Vertical);
    // cv.flip(src, src, FlipType.Horizontal);
    cv.rotate(src, src, cv.ROTATE_180);
    return src;
  }
  static MatRotateClockWise90(src) {
    cv.rotate(src, src, cv.ROTATE_90_COUNTERCLOCKWISE);
    return src;
  }
  static ShowMat(mat,id) {
    const s = document.createElement("canvas");
    document.querySelector("#canvasBox").appendChild(s);
    s.id = id;
    cv.imshow(s, mat);
    return s;
  }
}