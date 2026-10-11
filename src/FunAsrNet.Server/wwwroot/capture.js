// 浏览器只负责采集；重采样、语音活动检测和推理均由 C# 服务完成。
class PcmCapture extends AudioWorkletProcessor {
  /** 初始化一个 20 ms 的 PCM16 采集缓冲区及停止消息处理。 */
  constructor() {
    super();
    this.size = Math.round(sampleRate * 0.02);
    this.buffer = new ArrayBuffer(this.size * 2);
    this.view = new DataView(this.buffer);
    this.count = 0;
    this.peak = 0;
    this.active = true;
    this.port.onmessage = (e) => {
      if (e.data === "stop") {
        this.active = false;
        this.flush();
        this.port.postMessage({ stopped: true });
      }
    };
  }
  /** 将有效 PCM 缓冲区的所有权转交主线程，尾部不足一帧时仅复制有效区。 */
  flush() {
    if (!this.count) return;
    const bytes =
      this.count === this.size
        ? this.buffer
        : this.buffer.slice(0, this.count * 2);
    this.port.postMessage({ bytes, peak: this.peak }, [bytes]);
    this.buffer = new ArrayBuffer(this.size * 2);
    this.view = new DataView(this.buffer);
    this.count = 0;
    this.peak = 0;
  }
  /**
   * 将浏览器输入混合为单声道，限幅后写入小端 PCM16。
   * @param inputs 按输入节点、声道、采样排列的音频数组。
   * @returns 停止后返回 false，否则继续接收音频。
   */
  process(inputs) {
    if (!this.active) return false;
    const channels = inputs[0];
    if (channels?.length)
      for (let i = 0; i < channels[0].length; i++) {
        let x = 0;
        for (const channel of channels) x += channel[i];
        x = Math.max(-1, Math.min(1, x / channels.length));
        this.peak = Math.max(this.peak, Math.abs(x));
        this.view.setInt16(
          this.count++ * 2,
          Math.round(x < 0 ? x * 32768 : x * 32767),
          true,
        );
        if (this.count === this.size) this.flush();
      }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
