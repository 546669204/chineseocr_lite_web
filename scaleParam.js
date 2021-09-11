export default class ScaleParam {
  constructor(
    srcWidth,
    srcHeight,
    dstWidth,
    dstHeight,
    scaleWidth,
    scaleHeight
  ) {
    this.srcWidth = srcWidth;
    this.srcHeight = srcHeight;
    this.dstWidth = dstWidth;
    this.dstHeight = dstHeight;
    this.scaleWidth = scaleWidth;
    this.scaleHeight = scaleHeight;
  }
  static GetScaleParam(src, dstSize) {
    let srcWidth, srcHeight, dstWidth, dstHeight;
    srcWidth = src.cols;
    dstWidth = src.cols;
    srcHeight = src.rows;
    dstHeight = src.rows;
   
    let scale = 1.0;
    if (dstWidth > dstHeight) {
      dstWidth = srcWidth * dstSize / srcHeight;
      dstHeight = dstSize;
    } else {
      dstWidth = dstSize;
      dstHeight = srcHeight * dstSize / srcWidth;
    }
    if (dstWidth % 32 != 0) {
      dstWidth = (parseInt(dstWidth / 32) + 1) * 32;
      dstWidth = Math.max(dstWidth, 32);
    }
    if (dstHeight % 32 != 0) {
      dstHeight = (parseInt(dstHeight / 32) + 1) * 32;
      dstHeight = Math.max(dstHeight, 32);
    }
    let scaleWidth = dstWidth / srcWidth;
    let scaleHeight = dstHeight / srcHeight;
    return new ScaleParam(
      srcWidth,
      srcHeight,
      dstWidth,
      dstHeight,
      scaleWidth,
      scaleHeight
    );
  }
}