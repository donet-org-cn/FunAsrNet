# 部署与 HTTPS

框架依赖发布需要目标机器安装 ASP.NET Core 10 运行时；开发和构建用 .NET 10 SDK。

```shell
dotnet publish src/FunAsrNet.Server -c Release -o artifacts/server
dotnet artifacts/server/FunAsrNet.Server.dll --urls http://localhost:18080 --models models --lexicon assets/mandarin-lexicon.json --threads 4
```

路径相对启动工作目录。准备 [固定模型](models.md) 后，服务加载/预热 ASR、VAD、Huayan；AISHELL3 存在时初始化多发音人。默认纯 C# CPU，GPU 暂需使用方显式配置。

## Linux 服务

发布文件放 `/opt/funasrnet/app`、词典放 `/opt/funasrnet/assets`、模型放 `/opt/funasrnet/models`。调整 `deploy/funasrnet.service` 的 dotnet 路径再安装：

```shell
sudo cp deploy/funasrnet.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now funasrnet
```

模板显式监听 `0.0.0.0:18080`，应用默认只监听 localhost。示例无账号鉴权，对外接入由业务网关控制。模板使用 DynamicUser，不固定 CPU 编号。确认 RK3588 拓扑后可设 `CPUAffinity=4 5 6 7` 与 `FUNASRNET_PIN_THREADS=1`，其他设备不可照搬。

## 远程录音

localhost 之外浏览器麦克风需 HTTPS，可用受信任反向代理证书，或测试证书：

```shell
dotnet artifacts/server/FunAsrNet.Server.dll --models models --lexicon assets/mandarin-lexicon.json --demoHttpPort 18080 --demoHttpsPort 18443 --certificate artifacts/demo.pfx --certificateIp 192.0.2.10
```

替换文档示例地址 `192.0.2.10` 为自己的 IPv4。证书生成后复用，浏览器需信任。这些端口配置覆盖 `--urls`，PFX 含私钥不可入库。

默认 4 个音频连接、ASR/TTS 各一个上下文；增加前测量内存、吞吐和尾延迟。详见 [接口](api.md) 与 [并发](concurrency-and-memory.md)。
