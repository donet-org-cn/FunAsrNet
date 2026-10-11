using System.Text;
using System.Text.RegularExpressions;

namespace FunAsrNet;

/// <summary>
/// Qwen 字节级 BPE 分词器，包含聊天协议的特殊 token。
/// </summary>
public sealed class ByteBpe
{
    private readonly string[] _tokens;
    private readonly Dictionary<string, int> _ids = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> _merges = new();
    private readonly Dictionary<char, byte> _decode = new();
    private readonly char[] _encode = new char[256];
    private readonly Regex _special;
    private static readonly Regex Pieces = new(
        @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.CultureInvariant);
    private readonly HashSet<int> _ends = [];
    /// <summary>
    /// 判断 token 是否表示识别结束或聊天轮次结束。
    /// </summary>
    /// <param name="id">待检查的 token 编号。</param>
    /// <returns>满足所检查条件时为 true，否则为 false。</returns>
    public bool IsEnd(int id) => _ends.Contains(id);
    /// <summary>
    /// Qwen 字节级 BPE 分词器，包含聊天协议的特殊 token。
    /// </summary>
    /// <param name="model">已加载的模型及其权重；发布给并发上下文后不得修改权重。</param>
    public ByteBpe(Gguf model)
    {
        _tokens = ((object[])model.Metadata["tokenizer.ggml.tokens"]).Cast<string>().ToArray();
        for (int i = 0; i < _tokens.Length; ++i)
        {
            _ids[_tokens[i]] = i;
        }

        var merges = (object[])model.Metadata["tokenizer.ggml.merges"];
        for (int i = 0; i < merges.Length; ++i)
        {
            string merge = (string)merges[i];
            int split = merge.IndexOf(' ');
            _merges.TryAdd((merge[..split], merge[(split + 1)..]), i);
        }

        int extra = 0;
        for (int b = 0; b < 256; ++b)
        {
            char ch = (char)(b is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255 ? b : 256 + extra++);
            _encode[b] = ch;
            _decode[ch] = (byte)b;
        }

        string[] special = _tokens.Where(x => x.StartsWith("<|", StringComparison.Ordinal) && x.EndsWith("|>", StringComparison.Ordinal)).OrderByDescending(x => x.Length).ToArray();
        _special = new Regex("(" + string.Join('|', special.Select(Regex.Escape)) + ")", RegexOptions.CultureInvariant);
        foreach (string name in new[]
            {
                "<|endoftext|>",
                "<|im_end|>",
                "<|fim_suffix|>"
            }
        )
        {
            if (_ids.TryGetValue(name, out int id))
            {
                _ends.Add(id);
            }
        }

        _ends.Add(model.Int("tokenizer.ggml.eos_token_id", 151645));
    }

    /// <summary>
    /// 将文本转换为字节级 BPE token 序列。
    /// </summary>
    /// <param name="text">待编码、识别或合成的文本。</param>
    /// <returns>按输出顺序排列的整数数组。</returns>
    public int[] Encode(string text)
    {
        List<int> result = [];
        foreach (string chunk in _special.Split(text))
        {
            if (chunk.Length == 0)
            {
                continue;
            }

            if (chunk.StartsWith("<|", StringComparison.Ordinal) && _ids.TryGetValue(chunk, out int special))
            {
                result.Add(special);
                continue;
            }

            foreach (Match match in Pieces.Matches(chunk))
            {
                string encoded = new(Encoding.UTF8.GetBytes(match.Value).Select(b => _encode[b]).ToArray());
                if (_ids.TryGetValue(encoded, out int whole))
                {
                    result.Add(whole);
                    continue;
                }

                List<string> parts = encoded.Select(c => c.ToString()).ToList();
                while (parts.Count > 1)
                {
                    int best = int.MaxValue, at = -1;
                    for (int i = 0; i < parts.Count - 1; ++i)
                    {
                        if (_merges.TryGetValue((parts[i], parts[i + 1]), out int rank) && rank < best)
                        {
                            best = rank;
                            at = i;
                        }
                    }

                    if (at < 0)
                    {
                        break;
                    }

                    parts[at] += parts[at + 1];
                    parts.RemoveAt(at + 1);
                }

                foreach (string p in parts)
                {
                    result.Add(_ids[p]);
                }
            }
        }

        return result.ToArray();
    }

    /// <summary>
    /// 取得 token 对应的 UTF-8 字节内容。
    /// </summary>
    /// <param name="token">待解码的 token 编号。</param>
    /// <returns>调用成功时返回的结果；失败通过异常报告。</returns>
    public byte[] DecodeBytes(int token)
    {
        string piece = _tokens[token];
        if (piece.StartsWith("<|", StringComparison.Ordinal))
        {
            return Encoding.UTF8.GetBytes(piece);
        }

        return piece.Select(c => _decode[c]).ToArray();
    }

    /// <summary>
    /// 将 token 序列还原为文本并处理结束标记。
    /// </summary>
    /// <param name="tokens">按序排列的 token ID。</param>
    /// <returns>完成处理后的文本。</returns>
    public string Decode(IEnumerable<int> tokens) => Encoding.UTF8.GetString(tokens.SelectMany(DecodeBytes).ToArray());
}
