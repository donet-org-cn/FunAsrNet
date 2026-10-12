namespace FunAsrNet.Checks;

/// <summary>在真实 JIT 与 AOT 进程中验证中文词典的源生成反序列化与音素数组。</summary>
internal static class FrontendJsonChecks
{
    /// <summary>从带中文及转义键的 JSON 文件构造前端，并校验变调与完整编码。</summary>
    /// <param name="check">接收检查结果和错误说明的断言回调。</param>
    internal static void Run(Action<bool, string> check)
    {
        string lexicon = Path.GetTempFileName();
        string config = Path.GetTempFileName();
        try
        {
            File.WriteAllText(lexicon, """{"\u4f60":"ni3","好":"hao3","你好":"ni2 hao3"}""");
            File.WriteAllText(config, """{"phoneme_id_map":{"^":[1],"_":[0],"$":[2],"n":[4,40],"i":[5],"2":[6]," ":[7],"h":[8],"a":[9],"o":[10],"3":[11]}}""");
            var frontend = new MandarinFrontend(lexicon, config);
            check(frontend.Phonemize("你好") == "ni2 hao3", "Source-generated Chinese lexicon phrase");
            check(frontend.Phonemize("你 好") == "ni2 hao3", "Source-generated escaped lexicon key and sandhi");
            check(frontend.Encode("你好").SequenceEqual(new[] { 1, 0, 4, 40, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 2 }), "Source-generated phoneme map with multi-ID symbols");
        }
        finally
        {
            File.Delete(lexicon);
            File.Delete(config);
        }
    }
}
