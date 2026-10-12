const $ = (id) => document.getElementById(id);
const seconds = (n) => `${Number(n).toFixed(2)} 秒`;
const tunnel = document.getElementById("tunnel-command");
if (tunnel)
  tunnel.textContent = `ssh -N -L 18080:127.0.0.1:18080 <用户名>@${location.hostname}`;
let current = null,
  wavBusy = false,
  ttsBusy = false,
  ttsUrl = null;
/**
 * 更新状态文本及错误样式。
 * @param id 目标元素编号
 * @param message 要展示的消息
 * @param error 是否显示错误样式
 */
function status(id, message, error = false) {
  $(id).textContent = message;
  $(id).classList.toggle("error", error);
}
/**
 * 切换当前示例面板。
 * @param id 面板编号：live、upload 或 tts
 */
function choosePanel(id) {
  if (!["live", "upload", "tts"].includes(id)) id = "live";
  for (const panel of document.querySelectorAll(".panel"))
    panel.hidden = panel.id !== id;
  for (const tab of document.querySelectorAll(".tab")) {
    const selected = tab.dataset.panel === id;
    tab.classList.toggle("selected", selected);
    if (selected) tab.setAttribute("aria-current", "page");
    else tab.removeAttribute("aria-current");
  }
}
for (const tab of document.querySelectorAll(".tab"))
  tab.onclick = () => {
    location.hash = tab.dataset.panel;
  };
window.addEventListener("hashchange", () =>
  choosePanel(location.hash.slice(1)),
);
choosePanel(location.hash.slice(1));
/**
 * 刷新设备在线状态和推理忙闲信息。
 */
async function health() {
  try {
    const r = await fetch("/health", { cache: "no-store" });
    if (!r.ok) throw Error();
    const h = await r.json();
    status("health", `● 设备在线 · ${h.asrBusy ? "识别中" : "已就绪"}`);
    $("health").classList.add("ok");
  } catch {
    status("health", "设备暂未连接", true);
    $("health").classList.remove("ok");
  }
}
health();
setInterval(health, 10000);
const secure = window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;
if (!secure) {
  $("secure-note").hidden = false;
  fetch("/demo/config")
    .then((r) => r.json())
    .then((c) => {
      if (c.httpsPort) {
        const link = new URL(location.href);
        link.protocol = "https:";
        link.port = c.httpsPort;
        link.hash = "live";
        $("https-link").href = link.href;
        $("https-link").hidden = false;
      }
    })
    .catch(() => {});
}
/**
 * 按本页面当前操作更新按钮状态，避免重复提交。
 */
function buttons() {
  $("mic-start").disabled = !secure || !!current || wavBusy;
  $("sample-start").disabled = !!current || wavBusy;
  $("live-stop").disabled = !current || current.stopping;
  $("wav-submit").disabled =
    wavBusy || !!current || !$("wav-file").files.length;
}
buttons();
/**
 * 提取服务错误说明。
 * @param r HTTP 响应对象
 */
async function responseError(r) {
  if (r.status === 429) return "会话或推理队列已满，请稍后再试。";
  if (r.status === 413) return "文件超过 32 MB，请使用较短的录音。";
  const text = await r.text();
  try {
    const data = JSON.parse(text);
    return typeof data === "string" ? data : data.detail || data.title || text;
  } catch {
    return text.slice(0, 300) || `请求失败（${r.status}）`;
  }
}
/**
 * 发送 PCM16 帧，发送积压时结束当前会话。
 * @param s 当前会话状态
 * @param bytes 可发送的音频字节视图
 */
function sendAudio(s, bytes) {
  if (s !== current || s.ws?.readyState !== WebSocket.OPEN) return;
  if (s.ws.bufferedAmount > 128 * 1024) {
    fail(s, "网络发送积压，已停止录音，请检查连接后重试。");
    return;
  }
  s.ws.send(bytes);
  s.sent += bytes.byteLength / 2 / s.rate;
  $("live-clock").textContent =
    `${String(Math.floor(s.sent / 60)).padStart(2, "0")}:${String(Math.floor(s.sent % 60)).padStart(2, "0")}`;
}
/**
 * 停止采集并释放音频节点和媒体设备。
 * @param s 需要释放的会话
 */
async function releaseAudio(s) {
  clearTimeout(s.sampleTimer);
  s.stream?.getTracks().forEach((t) => t.stop());
  s.source?.disconnect();
  s.node?.disconnect();
  s.gain?.disconnect();
  if (s.ctx && s.ctx.state !== "closed") await s.ctx.close().catch(() => {});
}
/**
 * 完成会话并恢复页面交互。
 * @param s 需要结束的会话
 */
async function finish(s) {
  clearTimeout(s.endTimer);
  await releaseAudio(s);
  if (current !== s) return;
  current = null;
  $("live-dot").classList.remove("active");
  $("level").value = 0;
  buttons();
  health();
}
/**
 * 记录会话错误并释放资源。
 * @param s 失败的会话
 * @param message 用户可读的错误说明
 */
function fail(s, message) {
  if (s !== current || s.failed) return;
  s.failed = true;
  status("live-status", message, true);
  s.ws?.close();
  finish(s);
}
/**
 * 连接实时转写接口，收到 ready 后才允许音频输入。
 * @param s 维护套接字、采样率及结果的会话状态
 */
async function connect(s) {
  const endpoint = new URL("/asr/stream", location.href);
  endpoint.protocol = location.protocol === "https:" ? "wss:" : "ws:";
  endpoint.search = new URLSearchParams({
    codec: "pcm16",
    rate: s.rate,
    partials: "true",
    silenceMs: "900",
  });
  s.ws = new WebSocket(endpoint);
  s.ws.binaryType = "arraybuffer";
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(Error("连接超时"));
      s.ws.close();
    }, 15000);
    s.ws.onerror = () => {
      clearTimeout(timeout);
      reject(Error("连接失败，设备可能正忙或暂时离线。"));
      fail(s, "连接失败，设备可能正忙或暂时离线。");
    };
    s.ws.onclose = () => {
      clearTimeout(timeout);
      reject(Error("连接已关闭"));
      if (current === s && !s.done && !s.failed)
        fail(s, "连接中断，请重新开始。");
    };
    s.ws.onmessage = (event) => {
      if (current !== s) return;
      const data = JSON.parse(event.data);
      if (data.event === "ready") {
        clearTimeout(timeout);
        resolve();
        return;
      }
      if (data.event === "partial" || data.event === "final") {
        if (data.text) $("live-placeholder").hidden = true;
        if (data.event === "partial") $("live-partial").textContent = data.text;
        else {
          s.final.push(data.text);
          $("live-final").textContent = s.final.filter(Boolean).join("\n");
          $("live-partial").textContent = "";
        }
        $("live-metrics").textContent =
          `最近${data.event === "partial" ? "预览" : "完整句"}识别耗时 ${seconds(data.inferenceSeconds)} · 音频 ${Number(data.audioStartSeconds).toFixed(1)}–${Number(data.audioEndSeconds).toFixed(1)} 秒`;
      }
      if (data.event === "done") {
        s.done = true;
        $("live-partial").textContent = "";
        status(
          "live-status",
          s.final.some(Boolean) ? "转写完成" : "已结束，未检测到可识别语音",
        );
        if (s.stopTime)
          $("live-metrics").textContent +=
            ` · 停止后完成 ${seconds((performance.now() - s.stopTime) / 1000)}`;
        s.ws.close(1000, "finished");
        finish(s);
      }
    };
  });
}
/**
 * 启动录音或按真实时间发送标准样例。
 * @param sample 为 true 时使用样例，为 false 时使用麦克风
 */
async function begin(sample) {
  if (current || wavBusy) return;
  const s = { sent: 0, final: [], stopping: true, rate: 16000 };
  current = s;
  $("live-final").textContent = "";
  $("live-partial").textContent = "";
  $("live-placeholder").hidden = false;
  $("live-clock").textContent = "00:00";
  status("live-status", sample ? "正在加载样例…" : "正在申请麦克风权限…");
  buttons();
  try {
    let bytes;
    if (sample) {
      const r = await fetch("/sample.wav");
      if (!r.ok) throw Error("无法加载样例");
      const wav = await r.arrayBuffer();
      bytes = new Uint8Array(wav, 44);
    } else {
      s.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
      if (current !== s) {
        await releaseAudio(s);
        return;
      }
      s.ctx = new AudioContext();
      s.rate = s.ctx.sampleRate;
      await s.ctx.resume();
      await s.ctx.audioWorklet.addModule("/capture.js");
      s.source = s.ctx.createMediaStreamSource(s.stream);
      s.node = new AudioWorkletNode(s.ctx, "pcm-capture");
      s.gain = s.ctx.createGain();
      s.gain.gain.value = 0;
      s.node.port.onmessage = (e) => {
        if (e.data.stopped) {
          s.flushed?.();
          return;
        }
        if (e.data.bytes) sendAudio(s, e.data.bytes);
        if (current === s) $("level").value = e.data.peak || 0;
      };
      s.stream.getAudioTracks()[0].addEventListener("ended", () => {
        if (current === s && !s.stopping) stop();
      });
    }
    await connect(s);
    if (current !== s) return;
    s.stopping = false;
    buttons();
    $("live-dot").classList.add("active");
    status(
      "live-status",
      sample ? "样例按实时速度发送中…" : "正在聆听，请说普通话…",
    );
    if (sample) {
      let offset = 0;
      const start = performance.now();
      const tick = () => {
        if (current !== s || s.stopping) return;
        if (offset >= bytes.length) {
          stop();
          return;
        }
        const n = Math.min(640, bytes.length - offset);
        sendAudio(s, bytes.slice(offset, offset + n));
        offset += n;
        s.sampleTimer = setTimeout(
          tick,
          Math.max(0, offset / 32 - (performance.now() - start)),
        );
      };
      tick();
    } else s.source.connect(s.node).connect(s.gain).connect(s.ctx.destination);
  } catch (e) {
    fail(
      s,
      e.name === "NotAllowedError"
        ? "麦克风权限被拒绝，请在浏览器地址栏允许录音后重试。"
        : e.name === "NotFoundError"
          ? "未找到麦克风，请连接麦克风后重试。"
          : e.message,
    );
  }
}
/**
 * 刷新末尾音频后发送结束消息，等待最终识别结果。
 */
async function stop() {
  const s = current;
  if (!s || s.stopping) return;
  s.stopping = true;
  s.stopTime = performance.now();
  buttons();
  status("live-status", "正在完成最后一句…");
  clearTimeout(s.sampleTimer);
  try {
    if (s.node)
      await new Promise((resolve, reject) => {
        const timer = setTimeout(
          () => reject(Error("录音设备未能完成最后一帧，请重试。")),
          3000,
        );
        s.flushed = () => {
          clearTimeout(timer);
          resolve();
        };
        s.node.port.postMessage("stop");
      });
    if (current === s && s.ws.readyState === WebSocket.OPEN) {
      s.ws.send(JSON.stringify({ event: "end" }));
      s.endTimer = setTimeout(
        () => fail(s, "等待识别结果超时，请重试。"),
        120000,
      );
    }
    await releaseAudio(s);
  } catch (e) {
    fail(s, e.message);
  }
}
$("mic-start").onclick = () => begin(false);
$("sample-start").onclick = () => begin(true);
$("live-stop").onclick = stop;
window.addEventListener("pagehide", () => {
  if (current) {
    current.ws?.close();
    releaseAudio(current);
  }
});
$("wav-file").onchange = () => {
  const file = $("wav-file").files[0];
  $("file-name").textContent = file
    ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB`
    : "最长 60 秒 · 最大 32 MB";
  status("wav-status", "选择文件后开始");
  buttons();
};
$("wav-submit").onclick = async () => {
  const file = $("wav-file").files[0];
  if (!file || current || wavBusy) return;
  if (file.size > 32 * 1024 * 1024) {
    status("wav-status", "文件超过 32 MB，请使用较短的录音。", true);
    return;
  }
  wavBusy = true;
  buttons();
  status("wav-status", "上传并识别中…");
  $("wav-text").textContent = "";
  const started = performance.now();
  try {
    const r = await fetch("/asr/wav", {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: file,
    });
    if (!r.ok) throw Error(await responseError(r));
    const data = await r.json();
    $("wav-text").textContent = data.text || "未识别到文本";
    $("wav-text").classList.remove("placeholder");
    status("wav-status", "转写完成");
    $("wav-metrics").textContent =
      `音频 ${seconds(data.audioSeconds)} · 识别 ${seconds(data.inferenceSeconds)} · RTF ${Number(data.realTimeFactor).toFixed(3)} · 含上传总耗时 ${seconds((performance.now() - started) / 1000)}`;
  } catch (e) {
    status("wav-status", e.message, true);
  } finally {
    wavBusy = false;
    buttons();
    health();
  }
};
/**
 * 更新合成文本的字符数量提示。
 */
function count() {
  $("char-count").textContent = `${$("tts-text").value.length} / 200`;
}
count();
$("tts-text").oninput = count;

let ttsSpeakerId = null;
/** 从服务读取实际安装的音色，避免示例展示不存在的发音人。 */
async function loadVoices() {
  $("tts-submit").disabled = true;
  try {
    const response = await fetch("/tts/voices");
    if (!response.ok) throw Error(await responseError(response));
    const metadata = await response.json();
    const select = $("tts-voice");
    select.replaceChildren();
    for (const voice of metadata.voices) {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = voice.name;
      select.append(option);
    }
    select.value = metadata.voices.some((voice) => voice.id === "aishell3")
      ? "aishell3"
      : metadata.defaults.voiceId;
    if (!select.value) throw Error("服务没有可用的发音人。");
    const rememberedSpeakers = new Map();
    let currentVoice;
    /** 更新列表选中样式和合成发音人提示，不触发合成请求。 */
    function updateSpeakerSelection() {
      const speaker = currentVoice.speakers.find(
        (item) => item.id === ttsSpeakerId,
      );
      rememberedSpeakers.set(currentVoice.id, ttsSpeakerId);
      for (const input of $("tts-speaker").querySelectorAll("input")) {
        input.checked = Number(input.value) === ttsSpeakerId;
        input.parentElement.classList.toggle("selected", input.checked);
      }
      $("tts-speaker-selected").textContent = speaker
        ? `当前选择：${speaker.name}（编号 ${speaker.id}）`
        : "当前音色库没有可用的发音人。";
      $("tts-submit").disabled = ttsBusy || !speaker;
    }
    /** 按真实名称和编号筛选可见列表，保留当前合成发音人。 */
    function renderSpeakers() {
      const query = $("tts-speaker-search").value.trim().toLocaleLowerCase();
      const matches = currentVoice.speakers.filter((speaker) =>
        `${speaker.id} ${speaker.name} ${speaker.sourceId ?? ""}`
          .toLocaleLowerCase()
          .includes(query),
      );
      const fragment = document.createDocumentFragment();
      for (const speaker of matches) {
        const label = document.createElement("label");
        label.className = "speaker-option";
        const input = document.createElement("input");
        input.type = "radio";
        input.name = "tts-speaker";
        input.value = speaker.id;
        input.onchange = () => {
          ttsSpeakerId = speaker.id;
          updateSpeakerSelection();
        };
        const name = document.createElement("span");
        name.textContent = speaker.name;
        label.append(input, name);
        fragment.append(label);
      }
      $("tts-speaker").replaceChildren(fragment);
      $("tts-speaker-count").textContent = query
        ? `${matches.length} / ${currentVoice.speakers.length} 位匹配`
        : `共 ${currentVoice.speakers.length} 位 · 点击选择`;
      $("tts-speaker-empty").hidden = matches.length > 0;
      updateSpeakerSelection();
    }
    /** 根据音色库刷新真实发音人，恢复该库上一次选择。 */
    function updateSpeakers() {
      const voice = metadata.voices.find((item) => item.id === select.value);
      currentVoice = voice;
      const preferred =
        rememberedSpeakers.get(voice.id) ?? voice.defaultSpeakerId;
      ttsSpeakerId =
        voice.speakers.find((speaker) => speaker.id === preferred)?.id ??
        voice.speakers[0]?.id ??
        null;
      $("tts-speaker-search").value = "";
      $("tts-speaker-search").disabled = voice.speakers.length < 2;
      $("tts-rate").value = String(voice.sampleRate);
      $("tts-voice-info").textContent =
        `${voice.speakers.length} 位发音人 · 原生 ${voice.sampleRate.toLocaleString()} Hz` +
        (voice.sampleRate === 8000
          ? " · 适合电话，调高输出采样率不会增加原有音质。"
          : "");
      $("tts-speaker-prev").disabled = voice.speakers.length < 2;
      $("tts-speaker-next").disabled = voice.speakers.length < 2;
      renderSpeakers();
    }
    /**
     * 按完整列表切换发音人，清除搜索以展示选中项。
     * @param offset 切换方向，上一位为 -1，下一位为 1。
     */
    function changeSpeaker(offset) {
      const speakers = currentVoice.speakers;
      if (!speakers.length) return;
      const index = speakers.findIndex(
        (speaker) => speaker.id === ttsSpeakerId,
      );
      ttsSpeakerId =
        speakers[(index + offset + speakers.length) % speakers.length].id;
      $("tts-speaker-search").value = "";
      renderSpeakers();
      $("tts-speaker")
        .querySelector("input:checked")
        ?.scrollIntoView({ block: "nearest" });
    }
    $("tts-speaker-search").oninput = renderSpeakers;
    $("tts-speaker-prev").onclick = () => changeSpeaker(-1);
    $("tts-speaker-next").onclick = () => changeSpeaker(1);
    select.onchange = updateSpeakers;
    updateSpeakers();
  } catch (error) {
    $("tts-speaker-count").textContent = "发音人加载失败";
    status("tts-status", error.message, true);
  }
}
loadVoices();
$("tts-submit").onclick = async () => {
  if (ttsBusy) return;
  const text = $("tts-text").value.trim();
  if (!text) {
    status("tts-status", "请先输入朗读文本。", true);
    return;
  }
  ttsBusy = true;
  $("tts-submit").disabled = true;
  status("tts-status", "正在生成语音…");
  $("tts-player").pause();
  $("tts-player").removeAttribute("src");
  $("tts-player").load();
  $("tts-download").hidden = true;
  if (ttsUrl) {
    URL.revokeObjectURL(ttsUrl);
    ttsUrl = null;
  }
  try {
    const request = {
      text,
      speed: Number($("tts-speed").value),
      sampleRate: Number($("tts-rate").value),
      format: "wav",
      voiceId: $("tts-voice").value,
      speakerId: ttsSpeakerId,
      volume: Number($("tts-volume").value),
      seed: Number($("tts-seed").value),
      noiseScale: Number($("tts-noise").value),
      durationNoise: Number($("tts-duration-noise").value),
    };
    const invalid = document.querySelector("#tts input:invalid");
    if (invalid) {
      invalid.reportValidity();
      throw Error("请检查高级参数的取值范围。");
    }
    const r = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!r.ok) throw Error(await responseError(r));
    const blob = await r.blob();
    ttsUrl = URL.createObjectURL(blob);
    $("tts-player").src = ttsUrl;
    $("tts-download").href = ttsUrl;
    $("tts-download").hidden = false;
    status(
      "tts-status",
      `生成完成 · ${request.voiceId} / ${request.speakerId} 号，点击播放试听`,
    );
    $("tts-metrics").textContent =
      `生成耗时 ${seconds(r.headers.get("X-Inference-Seconds"))} · 音频时长 ${seconds(r.headers.get("X-Audio-Seconds"))} · ${(blob.size / 1024).toFixed(1)} KB`;
  } catch (e) {
    status("tts-status", e.message, true);
  } finally {
    ttsBusy = false;
    $("tts-submit").disabled = false;
  }
};
/**
 * 复制指定结果并显示操作状态。
 * @param id 结果元素编号
 * @param button 显示操作反馈的按钮
 */
async function copy(id, button) {
  const text = $(id).textContent;
  if (!text) return;
  try {
    if (navigator.clipboard && window.isSecureContext)
      await navigator.clipboard.writeText(text);
    else {
      const input = document.createElement("textarea");
      input.value = text;
      input.style.position = "fixed";
      input.style.opacity = 0;
      document.body.append(input);
      input.select();
      const ok = document.execCommand("copy");
      input.remove();
      if (!ok) throw Error();
    }
    button.textContent = "已复制";
  } catch {
    button.textContent = "请选中文字复制";
  }
  setTimeout(() => (button.textContent = "复制结果"), 1800);
}
$("live-copy").onclick = (e) => copy("live-final", e.target);
$("wav-copy").onclick = (e) => copy("wav-text", e.target);
