using System.Globalization;
using System.Text;

namespace FunAsrNet;

/// <summary>
/// AISHELL3 的只读拼音词典前端；共享词典时，每次编码仍拥有独立输出。
/// </summary>
public sealed class IcefallFrontend
{
    private readonly Dictionary<string, int[]> _lexicon = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _tokens = new(StringComparer.Ordinal);
    private readonly int _longest;
    private readonly int? _blank;

    /// <summary>
    /// 加载模型配套的音素表和词典，不调用外部拼音或分词运行库。
    /// </summary>
    /// <param name="lexiconPath">每行包含词条及其空格分隔音素的词典路径。</param>
    /// <param name="tokensPath">每行包含音素及整数编号的符号表路径。</param>
    /// <param name="metadata">模型声明的空白插入规则。</param>
    public IcefallFrontend(string lexiconPath, string tokensPath, IReadOnlyDictionary<string, string> metadata)
    {
        foreach (string line in File.ReadLines(tokensPath))
        {
            int separator = line.LastIndexOf(' ');
            if (separator < 1)
            {
                throw new InvalidDataException("音素表行缺少名称或编号。");
            }

            _tokens.Add(line[..separator], int.Parse(line[(separator + 1)..], CultureInfo.InvariantCulture));
        }

        foreach (string line in File.ReadLines(lexiconPath))
        {
            string[] fields = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (fields.Length > 1)
            {
                _lexicon.TryAdd(fields[0], fields.Skip(1).Select(phone => _tokens[phone]).ToArray());
            }
        }

        _longest = _lexicon.Keys.Max(word => word.Length);
        if (metadata.GetValueOrDefault("add_blank") == "1")
        {
            _blank = int.Parse(metadata.GetValueOrDefault("blank_id", "0"), CultureInfo.InvariantCulture);
        }
    }

    /// <summary>
    /// 最长词条优先编码中文，数字逐位朗读；不支持的文字明确报错。
    /// </summary>
    /// <param name="text">待朗读的简体中文、数字和常用标点。</param>
    /// <returns>符合模型词典及空白插入规则的音素编号。</returns>
    public int[] Encode(string text)
    {
        if (string.IsNullOrWhiteSpace(text) || text.Length > 200)
        {
            throw new ArgumentException("文本长度须为 1–200 字符。");
        }

        var normalized = new StringBuilder(text.Length);
        foreach (char value in text.Normalize(NormalizationForm.FormKC))
        {
            normalized.Append(value switch
            {
                >= '0' and <= '9' => "零一二三四五六七八九"[value - '0'],
                >= '０' and <= '９' => "零一二三四五六七八九"[value - '０'],
                _ => char.ToLowerInvariant(value)
            });
        }

        text = normalized.ToString();
        var ids = new List<int>(text.Length * 3);
        bool pronounced = false;
        if (_tokens.TryGetValue("sil", out int silence))
        {
            ids.Add(silence);
        }

        for (int offset = 0; offset < text.Length;)
        {
            char value = text[offset];
            if (char.IsWhiteSpace(value) || value is '“' or '”' or '「' or '」' or '(' or ')' or '（' or '）' or '"' or '\'')
            {
                offset++;
                continue;
            }

            if (value is ',' or '.' or '!' or '?' or ';' or ':' or '，' or '。' or '！' or '？' or '；' or '：' or '、' or '—' or '…')
            {
                if (_tokens.TryGetValue("#0", out int pause) && (ids.Count == 0 || ids[^1] != pause))
                {
                    ids.Add(pause);
                }

                offset++;
                continue;
            }

            int length = Math.Min(_longest, text.Length - offset);
            for (; length > 0; length--)
            {
                if (_lexicon.TryGetValue(text.Substring(offset, length), out int[]? phones))
                {
                    ids.AddRange(phones);
                    pronounced = true;
                    break;
                }
            }

            if (length == 0)
            {
                throw new ArgumentException($"当前音色词典不支持 U+{(int)value:X4}（{value}）。");
            }

            offset += length;
        }

        if (!pronounced)
        {
            throw new ArgumentException("文本不包含可朗读内容。");
        }

        if (_tokens.TryGetValue("sil", out silence))
        {
            ids.Add(silence);
            ids.Add(_tokens["eos"]);
        }

        if (_blank is not int blank)
        {
            return ids.ToArray();
        }

        var result = new int[ids.Count * 2 + 1];
        Array.Fill(result, blank);
        for (int index = 0; index < ids.Count; index++)
        {
            result[index * 2 + 1] = ids[index];
        }

        return result;
    }
}
