// Aggregate 16 kHz mono PCM in the audio thread so transport scheduling cannot
// change chunk boundaries. Transfer the final partial block before the stop ACK.
class FourthCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options.processorOptions.samples;
    this.buffer = new Int16Array(this.size);
    this.used = 0;
    this.active = true;
    this.port.onmessage = () => {
      this.active = false;
      if (this.used) this.send(this.buffer.slice(0, this.used));
      this.port.postMessage("stopped");
    };
  }

  send(samples) {
    this.port.postMessage(samples.buffer, [samples.buffer]);
  }

  process(inputs) {
    const channels = inputs[0];
    if (!this.active || !channels.length) return this.active;
    for (let i = 0; i < channels[0].length; i++) {
      let value = 0;
      for (const channel of channels) value += channel[i];
      this.buffer[this.used++] = Math.max(-32768, Math.min(32767, Math.round(value / channels.length * 32768)));
      if (this.used === this.size) {
        this.send(this.buffer);
        this.buffer = new Int16Array(this.size);
        this.used = 0;
      }
    }
    return true;
  }
}
registerProcessor("fourth-capture", FourthCapture);
