using System.Net;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;

namespace FunAsrNet.Server;
// 可选的本地 HTTPS 支持浏览器录音，证书完全使用 .NET 密码库生成。
/// <summary>
/// 为局域网浏览器示例配置可持久化的本地自签名 TLS 证书。
/// </summary>
internal static class DemoTls
{
    /// <summary>
    /// 配置示例 HTTP 与 HTTPS 监听端口及本地自签名证书。
    /// </summary>
    /// <param name="builder">ASP.NET Core 应用构建器。</param>
    public static void Configure(WebApplicationBuilder builder)
    {
        int? httpsPort = builder.Configuration.GetValue<int?>("demoHttpsPort");
        if (httpsPort is null)
        {
            return;
        }

        string path = builder.Configuration["certificate"] ?? throw new ArgumentException("--certificate is required with --demoHttpsPort");
        if (!File.Exists(path))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            using RSA key = RSA.Create(2048);
            var request = new CertificateRequest("CN=FunAsrNet local demo", key, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
            var names = new SubjectAlternativeNameBuilder();
            names.AddDnsName("localhost");
            names.AddIpAddress(IPAddress.Loopback);
            if (builder.Configuration["certificateIp"] is { Length: > 0 } certificateIp)
            {
                names.AddIpAddress(IPAddress.Parse(certificateIp));
            }
            request.CertificateExtensions.Add(names.Build());
            request.CertificateExtensions.Add(new X509BasicConstraintsExtension(false, false, 0, true));
            request.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.DigitalSignature | X509KeyUsageFlags.KeyEncipherment, true));
            request.CertificateExtensions.Add(new X509EnhancedKeyUsageExtension(new OidCollection { new("1.3.6.1.5.5.7.3.1") }, true));
            using var certificate = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddYears(2));
            File.WriteAllBytes(path, certificate.Export(X509ContentType.Pfx));
            if (OperatingSystem.IsLinux())
            {
                File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite);
            }
        }

        var tls = X509CertificateLoader.LoadPkcs12FromFile(path, null);
        builder.WebHost.ConfigureKestrel(options =>
            {
                options.ListenAnyIP(builder.Configuration.GetValue("demoHttpPort", 18080));
                options.ListenAnyIP(httpsPort.Value, listen => listen.UseHttps(tls));
            });
    }
}
