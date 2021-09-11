class Preload {
  loads = [{
      type: "script",
      url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.8.2-dev.20210831.0/dist/ort.min.js"
    }, {
      type: "script",
      url: "https://docs.opencv.org/4.5.3/opencv.js"
    }, {
      type: "script",
      url: "https://cdn.jsdelivr.net/npm/clipper-lib@6.4.2/clipper.js"
    },

    {
      type: "model",
      url: "/models/angle_net.onnx"
    }, {
      type: "model",
      url: "/models/crnn_lite_lstm.onnx"
    }, {
      type: "model",
      url: "/models/dbnet.onnx"
    },

    {
      type: "fetch",
      url: "/models/keys.txt"
    },
  ]
  constructor() {
    this.init();
  }
  init() {
    this.count = this.loads.length;
    this.done = 0;
    this.loads.map(it => this["pre_" + it.type](it).then(r=>{
      this.done++;
      document.querySelector("#loadding-box p ").innerHTML = `加载中... (${this.done}/${this.count})`;
      if(this.done >= this.count){
        document.querySelector("#loadding-box").remove();
        this.pre_module({url:"main.js"})
      }
    }));
  }
  pre_module(it){
    return new Promise(function(resolve) {
      const script = document.createElement("script");
      script.type = "module";
      script.async = true;
      script.onload = function() {
        resolve()
      };
      script.onerror = function(error){
        reject(error)
      }
      script.src = it.url;
      document.body.appendChild(script)
    })
  }
  pre_script(it) {
    return new Promise(function(resolve) {
      const script = document.createElement("script");
      script.type = "text/javascript";
      script.async = true;
      script.onload = function() {
        resolve()
      };
      script.onerror = function(error){
        reject(error)
      }
      script.src = it.url;
      document.body.appendChild(script)
    })
  }
  pre_model(it){
    return fetch(it.url)
  }
  pre_fetch(it){
    return fetch(it.url)
  }

}

new Preload();