export default class BaseModel {
  constructor(path) {
    this.initModel(path);
  }
  initModel(path) {
    this.modelLoading = ort.InferenceSession.create(path, {
      executionProviders: ["wasm"],
    }).then((s) => {
      this.session = s;
    });
  }
}