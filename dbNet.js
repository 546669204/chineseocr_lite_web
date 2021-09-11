import BaseModel from "./baseModel.js"
import OcrUtils from "./utils.js";

class TextBox {

}

export default class dbNetModel extends BaseModel {
  constructor() {
    super("./models/dbnet.onnx");
  }
  async run(src, scale, boxScoreThresh, boxThresh, unClipRatio) {
    await this.modelLoading;
    let srcResize = new cv.Mat();
    cv.resize(src, srcResize, new cv.Size(scale.dstWidth, scale.dstHeight));
    console.time("SubstractMeanNormalize")
    const input0 = OcrUtils.SubstractMeanNormalize(
      srcResize, [0.485 * 255, 0.456 * 255, 0.406 * 255], [1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0]
    );
    console.timeEnd("SubstractMeanNormalize")
    const feeds = {
      input0
    };
    console.time("dbNetModel")
    const results = await this.session.run(feeds);
    console.timeEnd("dbNetModel")
    try {
      return this.GetTextBoxes(
        results.out1.data,
        srcResize.rows,
        srcResize.cols,
        scale,
        boxScoreThresh,
        boxThresh,
        unClipRatio
      );
    } finally {
      srcResize.delete();
    }
  }
  LengthOfPoints(box) {
    let length = 0;

    let pt = box[0];
    let x0 = pt.x;
    let y0 = pt.y;
    let x1 = 0,
      y1 = 0,
      dx = 0,
      dy = 0;
    box.push(pt);

    let count = box.length;
    for (let idx = 1; idx < count; idx++) {
      let pts = box[idx];
      x1 = pts.x;
      y1 = pts.y;
      dx = x1 - x0;
      dy = y1 - y0;

      length += Math.sqrt(dx * dx + dy * dy);

      x0 = x1;
      y0 = y1;
    }

    box.splice(count - 1);
    return length;
  }
  SignedPolygonArea(Points) {
    // Add the first point to the end.
    let num_points = Points.length;
    let pts = Points.slice();
    // Points.CopyTo(pts, 0);
    pts[num_points] = Points[0];

    // Get the areas.
    let area = 0;
    for (let i = 0; i < num_points; i++) {
      area +=
        (pts[i + 1].x - pts[i].x) *
        (pts[i + 1].y + pts[i].y) / 2;
    }

    return area;
  }
  Unclip(box, unclip_ratio) {
    let theCliperPts = [];
    for (let pt of box) {
      let a1 = new ClipperLib.IntPoint(pt.x, pt.y);
      theCliperPts.push(a1);
    }

    let area = Math.abs(this.SignedPolygonArea(box));
    let length = this.LengthOfPoints(box);
    let distance = area * unclip_ratio / length;

    let co = new ClipperLib.ClipperOffset();
    co.AddPath(theCliperPts, ClipperLib.JoinType.jtRound, ClipperLib.EndType.etClosedPolygon);
    let solution = [];
    co.Execute(solution, distance);
    if (solution.length == 0) {
      return null;
    }

    let retPts = [];
    for (let ip of solution[0]) {
      retPts.push(ip.X, ip.Y);
    }
    return cv.matFromArray(retPts.length / 2, 1, cv.CV_32SC2, new Float32Array(retPts));
  }
  GetMiniBox(contours) {
    let box = [];
    let rrect = cv.minAreaRect(contours);
    let points = cv.RotatedRect.points(rrect);


    let thePoints = points.slice();
    thePoints.sort((left, right) => {
      if (left == null && right == null) {
        return 1;
      }

      if (left == null) {
        return 0;
      }

      if (right == null) {
        return 1;
      }

      if (left.x > right.x) {
        return 1;
      }

      if (left.x == right.x) {
        return 0;
      }

      return -1;
    });

    let index_1 = 0,
      index_2 = 1,
      index_3 = 2,
      index_4 = 3;
    if (thePoints[1].y > thePoints[0].y) {
      index_1 = 0;
      index_4 = 1;
    } else {
      index_1 = 1;
      index_4 = 0;
    }

    if (thePoints[3].y > thePoints[2].y) {
      index_2 = 2;
      index_3 = 3;
    } else {
      index_2 = 3;
      index_3 = 2;
    }

    box.push(thePoints[index_1]);
    box.push(thePoints[index_2]);
    box.push(thePoints[index_3]);
    box.push(thePoints[index_4]);

    return box;
  }
  GetScore(contour, fMapMat,contours,i) {
    let xmin = 9999;
    let ymin = 9999;

    let brect = cv.boundingRect(contour);
    xmin = brect.x;
    ymin = brect.y;

    let roiWidth = brect.width;
    let roiHeight = brect.height;

    let rect = new cv.Rect(xmin, ymin, roiWidth, roiHeight);

    let roiBitmap = fMapMat.roi(rect);

    let mask2 = new cv.Mat.zeros(fMapMat.rows, fMapMat.cols, cv.CV_8U);

    cv.drawContours(mask2, contours, i, new cv.Scalar(255, 255, 255), -1);
    let mask = mask2.roi(rect);
    mask2.delete();

    try {
      return cv.mean(roiBitmap, mask)[0];
    } catch (error) {
      vvp.delete();
      roiBitmap.delete();
      mask.delete();
    }
  }
  GetTextBoxes(data, rows, cols, s, boxScoreThresh, boxThresh, unClipRatio) {
    let minArea = 3.0;
    let rsBoxes = [];

    let outputData = data;
    let fMapMat = cv.matFromArray(rows, cols, cv.CV_32F, outputData);
    let norfMapMat2 = new cv.Mat();
    let norfMapMat = new cv.Mat();
    cv.threshold(fMapMat, norfMapMat2, boxThresh, 255, cv.THRESH_BINARY)
    norfMapMat2.convertTo(norfMapMat, cv.CV_8U)
    norfMapMat2.delete();
    let hierarchy = new cv.Mat();
    let contours = new cv.MatVector();
    cv.findContours(
      norfMapMat,
      contours,
      hierarchy,
      cv.RETR_LIST,
      cv.CHAIN_APPROX_SIMPLE
    );
    norfMapMat.delete();
    hierarchy.delete();

    for (let i = 0; i < contours.size(); i++) {
      let minEdgeSize = 0;
      let box = this.GetMiniBox(contours.get(i), minEdgeSize);
      let rrect = cv.minAreaRect(contours.get(i));
      minEdgeSize = Math.min(rrect.size.width, rrect.size.height);
      if (minEdgeSize < minArea) {
        continue;
      }
      let score = this.GetScore(contours.get(i), fMapMat,contours,i);
      if (score < boxScoreThresh) {
        continue;
      }
      let newBox = this.Unclip(box, unClipRatio);
      if (newBox == null) {
        continue;
      }

      let minBox = this.GetMiniBox(newBox, minEdgeSize);
      rrect = cv.minAreaRect(newBox);
      minEdgeSize = Math.min(rrect.size.width, rrect.size.height);

      if (minEdgeSize < minArea + 2) {
        continue;
      }
      let finalPoints = [];
      for (let item of minBox) {
        let x = parseInt(item.x / s.scaleWidth);
        let ptx = Math.min(Math.max(x, 0), s.srcWidth);

        let y = parseInt(item.y / s.scaleHeight);
        let pty = Math.min(Math.max(y, 0), s.srcHeight);
        let dstPt = new cv.Point(ptx, pty);
        finalPoints.push(dstPt);
      }

      let textBox = new TextBox();
      textBox.Score = score;
      textBox.Points = finalPoints;
      rsBoxes.push(textBox);
    }
    contours.delete();
    fMapMat.delete();
    rsBoxes.reverse();
    return rsBoxes;
  }
}