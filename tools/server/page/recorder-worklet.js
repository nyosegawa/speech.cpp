// The audio thread's side of the recorder: passes each block of the microphone's first channel to the page.

class Recorder extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0][0];
    if (channel) this.port.postMessage(channel.slice());
    return true;
  }
}

registerProcessor('recorder', Recorder);
