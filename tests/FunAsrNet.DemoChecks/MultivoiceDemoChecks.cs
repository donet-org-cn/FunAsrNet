using System.Net;
using System.Net.Http.Json;
using System.Text.Json;

namespace FunAsrNet.DemoChecks;

/// <summary>
/// 检查已安装多发音人服务的参数边界、并发选择和电话输出。
/// </summary>
internal static class MultivoiceDemoChecks
{
    /// <summary>
    /// 使用现有 HTTP 或证书固定的 HTTPS 客户端执行多发音人验收。
    /// </summary>
    /// <param name="http">连接目标服务的客户端。</param>
    /// <param name="cancellationToken">控制整批检查的取消标记。</param>
    internal static async Task RunAsync(HttpClient http, CancellationToken cancellationToken)
    {
        using var document = JsonDocument.Parse(await http.GetStringAsync("/tts/voices", cancellationToken));
        var voices = document.RootElement.GetProperty("voices").EnumerateArray().ToArray();
        var aishell = voices.SingleOrDefault(voice => voice.GetProperty("id").GetString() == "aishell3");
        if (aishell.ValueKind == JsonValueKind.Undefined)
        {
            Console.WriteLine("SKIP 未安装可选 AISHELL3 模型");
            return;
        }

        Check(aishell.GetProperty("speakerIds").EnumerateArray().Select(id => id.GetInt32()).SequenceEqual(Enumerable.Range(0, 174)), "174 个真实发音人编号");
        Check(aishell.GetProperty("speakers").GetArrayLength() == 174 && aishell.GetProperty("sampleRate").GetInt32() == 8000, "发音人详情与原生 8kHz");
        foreach (int speaker in new[] { -1, 174 })
        {
            using var response = await http.PostAsJsonAsync("/tts", new
            {
                text = "您好",
                voiceId = "aishell3",
                speakerId = speaker
            }, cancellationToken);
            Check(response.StatusCode == HttpStatusCode.BadRequest, $"拒绝 AISHELL3 越界编号 {speaker}");
        }

        var outputs = await Task.WhenAll(new[] { 50, 75, 150 }.Select(speaker => GenerateAsync(http, speaker, cancellationToken)));
        Check(!outputs[0].SequenceEqual(outputs[1]) && !outputs[1].SequenceEqual(outputs[2]), "并发请求使用不同发音人");
        byte[] repeated = await GenerateAsync(http, 50, cancellationToken);
        Check(outputs[0].SequenceEqual(repeated), "跨发音人请求保持确定性");
        await GenerateAsync(http, 173, cancellationToken);
        foreach (string format in new[] { "pcma", "pcmu", "pcm16" })
        {
            using var response = await http.PostAsJsonAsync("/tts/stream", new
            {
                text = "您好，欢迎致电。",
                voiceId = "aishell3",
                speakerId = 50,
                sampleRate = 8000,
                format
            }, cancellationToken);
            response.EnsureSuccessStatusCode();
            byte[] audio = await response.Content.ReadAsByteArrayAsync(cancellationToken);
            Check(audio.Length > 1600 && response.Headers.GetValues("X-Audio-Sample-Rate").Single() == "8000", $"多发音人流式 {format}");
        }
    }

    /// <summary>
    /// 生成同一句话并检查响应中的 WAV 文件标识与发音人选择。
    /// </summary>
    /// <param name="http">连接目标服务的客户端。</param>
    /// <param name="speaker">本次请求使用的发音人编号。</param>
    /// <param name="cancellationToken">控制请求等待的取消标记。</param>
    /// <returns>供确定性与音色差异检查的完整 WAV 数据。</returns>
    private static async Task<byte[]> GenerateAsync(HttpClient http, int speaker, CancellationToken cancellationToken)
    {
        using var response = await http.PostAsJsonAsync("/tts", new
        {
            text = "您好，欢迎致电，请问有什么可以帮您？",
            voiceId = "aishell3",
            speakerId = speaker,
            sampleRate = 8000,
            format = "wav"
        }, cancellationToken);
        response.EnsureSuccessStatusCode();
        byte[] audio = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        Check(audio.Length > 16000 && audio.AsSpan(0, 4).SequenceEqual("RIFF"u8), $"发音人 {speaker} WAV 输出");
        Console.WriteLine($"TTS aishell3/{speaker}: inference={response.Headers.GetValues("X-Inference-Seconds").Single()}s audio={response.Headers.GetValues("X-Audio-Seconds").Single()}s");
        return audio;
    }

    /// <summary>
    /// 输出成功检查，或在条件失败时终止验收。
    /// </summary>
    /// <param name="condition">必须满足的检查条件。</param>
    /// <param name="name">检查项目名称。</param>
    private static void Check(bool condition, string name)
    {
        if (!condition)
        {
            throw new InvalidDataException(name);
        }

        Console.WriteLine($"PASS {name}");
    }
}
