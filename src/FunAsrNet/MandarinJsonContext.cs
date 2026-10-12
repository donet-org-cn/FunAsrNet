using System.Text.Json.Serialization;

namespace FunAsrNet;

/// <summary>在编译时生成中文词典与音素表的 JSON 元数据，供 JIT 和 AOT 共用。</summary>
[JsonSerializable(typeof(Dictionary<string, string>), TypeInfoPropertyName = "Lexicon")]
[JsonSerializable(typeof(Dictionary<string, int[]>), TypeInfoPropertyName = "Symbols")]
internal partial class MandarinJsonContext : JsonSerializerContext
{
}
