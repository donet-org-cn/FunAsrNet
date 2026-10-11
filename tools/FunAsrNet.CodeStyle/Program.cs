using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace FunAsrNet.CodeStyle;

/// <summary>使用 SDK 自带 Roslyn 检查中文方法与参数注释、UTF-8、LF 和控制流大括号。</summary>
internal static class Program
{
    /// <summary>检查仓库内的 C# 源码；不读取生成目录，不依赖额外 NuGet 包。</summary>
    /// <param name="args">可选的仓库根目录，默认为当前工作目录。</param>
    /// <returns>全部通过时返回零；发现违反规范的源码时返回一。</returns>
    private static int Main(string[] args)
    {
        string root = Path.GetFullPath(args.Length > 0 ? args[0] : ".");
        int errors = 0;
        int methods = 0;
        int files = 0;
        foreach (string path in Directory.EnumerateFiles(root, "*.cs", SearchOption.AllDirectories))
        {
            if (path.Split(Path.DirectorySeparatorChar).Any(part => part is "bin" or "obj" or ".git" or "artifacts")
                || Path.GetRelativePath(root, path).Replace('\\', '/').StartsWith("src/FunAsrNet.Vulkan/Vendor/", StringComparison.Ordinal))
            {
                continue;
            }
            ++files;
            string source = File.ReadAllText(path, new UTF8Encoding(false, true));
            var tree = CSharpSyntaxTree.ParseText(source);
            if (source.Contains('\r') || source.Split('\n').Any(line => line.EndsWith(' ') || line.EndsWith('\t')))
            {
                Report(path, 1, "须使用 LF 且不能有行尾空白。", ref errors);
            }
            foreach (var node in tree.GetRoot().DescendantNodes())
            {
                var parameters = node switch
                {
                    BaseMethodDeclarationSyntax method => method.ParameterList.Parameters,
                    LocalFunctionStatementSyntax method => method.ParameterList.Parameters,
                    TypeDeclarationSyntax type => type.ParameterList?.Parameters ?? default,
                    _ => default
                };
                bool isMethod = node is BaseMethodDeclarationSyntax or LocalFunctionStatementSyntax;
                bool needsDoc = isMethod || node is BaseTypeDeclarationSyntax or PropertyDeclarationSyntax or EnumMemberDeclarationSyntax;
                if (isMethod)
                {
                    ++methods;
                }
                if (needsDoc)
                {
                    string comment = string.Concat(node.GetLeadingTrivia().Where(item => item.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) || isMethod && node is LocalFunctionStatementSyntax && item.IsKind(SyntaxKind.SingleLineCommentTrivia)).Select(item => item.ToFullString() + "\n"));
                    comment = Regex.Replace(comment, @"(?m)^\s*///? ?", "");
                    XElement? doc = null;
                    try
                    {
                        doc = XElement.Parse("<doc>" + comment + "</doc>");
                    }
                    catch (System.Xml.XmlException)
                    {
                        Report(path, Line(node), "XML 注释格式不正确。", ref errors);
                    }
                    if (!HasChinese(doc?.Element("summary")?.Value))
                    {
                        Report(path, Line(node), "缺少中文 summary。", ref errors);
                    }
                    foreach (var parameter in parameters)
                    {
                        string name = parameter.Identifier.Text;
                        var description = doc?.Elements("param").FirstOrDefault(item => (string?)item.Attribute("name") == name);
                        if (!HasChinese(description?.Value))
                        {
                            Report(path, Line(node), $"参数 {name} 缺少中文说明。", ref errors);
                        }
                    }
                }
                StatementSyntax? body = node switch
                {
                    IfStatementSyntax item => item.Statement,
                    ElseClauseSyntax { Statement: not IfStatementSyntax } item => item.Statement,
                    ForStatementSyntax item => item.Statement,
                    ForEachStatementSyntax item => item.Statement,
                    WhileStatementSyntax item => item.Statement,
                    DoStatementSyntax item => item.Statement,
                    UsingStatementSyntax item => item.Statement,
                    LockStatementSyntax item => item.Statement,
                    _ => null
                };
                if (body is not null and not BlockSyntax)
                {
                    Report(path, Line(node), "控制流语句必须使用大括号。", ref errors);
                }
            }
        }
        Console.WriteLine($"中文注释与格式检查：{files} 个文件，{methods} 个方法，{errors} 个错误。");
        return errors == 0 ? 0 : 1;
    }

    /// <summary>检查说明是否包含中文且不是未完成占位文本。</summary>
    /// <param name="text">XML 注释的纯文本。</param>
    /// <returns>包含实际中文说明时为 true。</returns>
    private static bool HasChinese(string? text)
    {
        return text is not null && Regex.IsMatch(text, @"[\u4e00-\u9fff]") && !text.Contains("待补充");
    }

    /// <summary>将 Roslyn 节点位置转换为从一开始的源码行号。</summary>
    /// <param name="node">待定位的语法节点。</param>
    /// <returns>适合编辑器跳转的行号。</returns>
    private static int Line(SyntaxNode node)
    {
        return node.GetLocation().GetLineSpan().StartLinePosition.Line + 1;
    }

    /// <summary>输出包含文件位置的错误并增加总数。</summary>
    /// <param name="path">源码路径。</param>
    /// <param name="line">从一开始的行号。</param>
    /// <param name="message">具体的规范问题。</param>
    /// <param name="errors">累计错误数。</param>
    private static void Report(string path, int line, string message, ref int errors)
    {
        ++errors;
        Console.Error.WriteLine($"{path}({line}): {message}");
    }
}
