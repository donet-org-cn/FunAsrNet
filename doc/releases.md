# NuGet 和 GitHub Release

正式版本的唯一来源是 Directory.Build.props 中的 Version。核心包 FunAsrNet 与可选 GPU 包 FunAsrNet.Vulkan 同版本发布；CLI、服务、测试和基准项目不打包为 NuGet 库。模型、字典、驱动和原生开发对照库不进入包。

## 发布流程

[release.yml](../.github/workflows/release.yml) 参考 [SimdSkia 的 NuGet 流水线](https://github.com/donet-org-cn/SimdSkia/blob/main/.github/workflows/nuget-preview.yml)，采用以下顺序：

1. Windows/Linux 编译、数值回归、标量回退、注释和格式检查。
2. Windows 生成两份 nupkg 和两份 snupkg，检查许可证、README、XML 文档、符号、版本和依赖关系，生成 SHA-256 清单。
3. Windows/Linux 从同一批冻结包建立独立包缓存，只用 PackageReference 安装并运行公开 API 检查。
4. 使用 GitHub OIDC 换取短期 NuGet 发布凭据，推送已测试的包及符号。
5. 从 NuGet 公开地址下载两个包，逐 ZIP 成员比对内容。NuGet 添加的 .signature.p7s 仓库签名单独排除，程序集、元数据等均须一致。
6. 核对成功后创建非预发布的 GitHub Release，并附带包、符号、清单和公开下载验证记录。

只支持正式 major.minor.patch。推送 v1.0.1 一类标签会发布；手工 workflow_dispatch 默认 publish=false，只验证。开启发布时，标签必须存在、指向本次运行的同一提交，且该提交位于 main 历史中。不要移动已经发布的标签或复用 NuGet 版本。

GitHub 托管 runner 不承诺有可用 GPU；GPU 设备数值验证与完整模型性能证据由本机报告提供，流水线不会将无 GPU 的安装测试写成 GPU 正确性测试。

## 一次性配置

仓库变量 NUGET_USER 使用 NuGet 的账号/配置文件名称，例如本组织既有流程的 dotnet.org.cn；不是邮箱。发布作业使用 GitHub environment nuget。

在该 NuGet 账号的 Trusted Publishing 页面创建匹配策略：

| 字段 | 值 |
|---|---|
| Repository Owner | donet-org-cn |
| Repository | FunAsrNet |
| Workflow File | release.yml |
| Environment | nuget |
| Package pattern | FunAsrNet* |
| Scope | 推送新包与现有包的新版本 |

策略所有者应为实际拥有这两个包的账号/组织。该仓库不存放长期 NuGet API key。配置含义见 [NuGet 官方 Trusted Publishing 文档](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing)。

## 本地验证

```powershell
dotnet build FunAsrNet.sln -c Release
dotnet run --project tests/FunAsrNet.Checks -c Release --no-build
./tools/PackRelease.ps1 -Version 1.0.1
./tools/TestPackages.ps1 -Version 1.0.1 -Packages artifacts/packages/1.0.1
```

若维护机器安装的 SDK 低于 global.json，而该源码已在相应 SDK 验证，可向两个打包脚本显式传入 -DotNetSdk 'C:/Program Files/dotnet/sdk/10.0.204/dotnet.dll'。正式 CI 使用 setup-dotnet 提供的 10.0.x，并遵循 global.json。

版本发布说明位于 doc/releases/<version>.md。完成上述验证后提交代码、推送 main，再创建与推送带注释的 v<version> 标签。流水线失败时检查失败步骤；已经推送到 NuGet 的版本不能用另一份内容覆盖。重复发布同一冻结包可跳过重复上传，但公开内容验证仍必须通过。
