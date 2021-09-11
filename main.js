import CRNNModel from "./crnnNet.js";
import dbNetModel from "./dbNet.js";
import angleNetModel from "./angleNet.js";
import ScaleParam from "./scaleParam.js";
import OcrUtils from "./utils.js";
// use an async context to call onnxruntime functions.
async function main(img, ctx) {


  let brgSrc = cv.imread(img); //default : BGR
  let originSrc = new cv.Mat();

  cv.cvtColor(brgSrc, originSrc, cv.COLOR_BGR2RGB); // convert to RGB
  let originRect = new cv.Rect(padding, padding, originSrc.cols, originSrc.rows);
  let paddingSrc = OcrUtils.MakePadding(originSrc, padding);
  if (paddingSrc != originSrc) {
    originSrc.delete();
  }
  let resize;
  if (imgResize <= 0) {
    resize = Math.min(paddingSrc.cols, paddingSrc.rows);
  } else {
    resize = imgResize;
  }
  let scale = ScaleParam.GetScaleParam(paddingSrc, resize);

  let src = paddingSrc.clone();
  paddingSrc.delete();
  // 开始
  let thickness = OcrUtils.GetThickness(src);
  console.time("dbNet")
  let textBoxes = await new dbNetModel().run(
    src,
    scale,
    boxScoreThresh,
    boxThresh,
    unClipRatio,
  );
  console.timeEnd("dbNet")

  let partImages = OcrUtils.GetPartImages(src, textBoxes);
  if (isPartImg) {
    for (let i = 0; i < partImages.length; i++) {
      OcrUtils.ShowMat(partImages[i]);
    }
  }
  console.time("angleNetModel")
  let angles = await new angleNetModel().run(partImages, doAngle, mostAngle);
  console.timeEnd("angleNetModel")
  console.log("angles", angles)
    //Rotate partImgs
  for (let i = 0; i < partImages.length; ++i) {
    if (angles[i].Index == 0) {
      partImages[i] = OcrUtils.MatRotateClockWise180(partImages[i]);
    }
    if (isDebugImg) {
      OcrUtils.ShowMat(partImages[i]);
    }
  }
  console.time("CRNNModel")
  let textLines = await new CRNNModel().run(partImages);
  console.timeEnd("CRNNModel")

  console.log(textLines)

  if (!ctx) {
    const el = OcrUtils.ShowMat(brgSrc);
    ctx = el.getContext("2d");
  } else {
    ctx.clearRect(0, 0, brgSrc.cols, brgSrc.rows)
  }

  function drawRect(points, text) {
    const color = `rgb(${Math.random()*255},${Math.random()*255},${Math.random()*255})`
    ctx.strokeStyle = color
    ctx.lineWidth = thickness;
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const element = points[i];
      if (i == 0) {
        ctx.moveTo(element.x, element.y);
      } else {
        ctx.lineTo(element.x, element.y);
      }
    }
    ctx.lineTo(points[0].x, points[0].y);
    ctx.stroke();
    ctx.fillStyle = color
    ctx.font = thickness * 8 + "px Arial"
    ctx.fillText(text, points[0].x, points[0].y)
  }
  textBoxes.forEach((it, i) => {
    drawRect(it.Points, textLines[i])
  })
}


const padding = 0;
const imgResize = 0;
const boxScoreThresh = 0.618;
const boxThresh = 0.3;
const minArea = 3;
const unClipRatio = 1.8;
const doAngle = true;
const mostAngle = false;
const isPartImg = false;
const isDebugImg = false;

(async function() {
  const input = document.querySelector("#file");
  input.addEventListener("change", function(e) {
    const file = e.target.files[0];

    const u = URL.createObjectURL(file);
    const i = new Image();
    i.onload = function() {
      URL.revokeObjectURL(u);
      document.querySelector("#canvasBox").innerHTML = ""
      main(i)
    };
    i.src = u;
  });

  document.querySelector("#playVideo").addEventListener("click", function() {
    navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: {
          exact: "environment"
        }
      },
      // video:true,
      audio: false
    }).then(function(stream) {
      document.querySelector("#canvasBox").innerHTML = "";
      const video = document.createElement("video");
      video.srcObject = stream;
      video.play();
      video.onloadeddata = function(){
        cap();
      }
      const canvasCapture = document.createElement("canvas");
      const canvasOutput = document.createElement("canvas");
      const ctxCapture = canvasCapture.getContext("2d");
      const ctxOutput = canvasOutput.getContext("2d");
      canvasOutput.id = "canvasOutput"

      async function cap() {
        canvasCapture.width = video.videoWidth;
        canvasCapture.height = video.videoHeight;
        if (canvasOutput.width != video.videoWidth) {
          canvasOutput.width = video.videoWidth;
          canvasOutput.height = video.videoHeight;
        }

        ctxCapture.drawImage(video, 0, 0);
        await main(canvasCapture, ctxOutput)
        setTimeout(() => {
          cap()
        }, 1)
      }
      document.querySelector("#canvasBox").appendChild(video);
      document.querySelector("#canvasBox").appendChild(canvasOutput);


    }).catch(function(err) {
      console.log(err);
    })
  })



})();