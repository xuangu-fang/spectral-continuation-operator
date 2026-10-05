# 科学备忘录的公式资产

对应文档：[谱可预测性、可辨识性与校准](../../SPECTRAL_PREDICTABILITY_IDENTIFIABILITY_CALIBRATION_ZH.md)。

为避免 GitHub／其他 Markdown 阅读器的数学宏兼容问题，正文的40条块公式使用本地 SVG。
它们是矢量图，不是截图；白色底板使公式在浅色／深色页面都可读，字形已转为路径，不依赖字体下载。
正文提供中文替代文字，完整可复制的 LaTeX 保存在 [equations.json](equations.json)。

修改方法：

1. 在 JSON 中编辑对应 `latex`。有 `lines` 时，它负责长公式的分行显示，也必须同步修改。
2. 在仓库根目录运行 `.venv/bin/python scripts/render_spectral_science_math.py`。
3. 运行 `.venv/bin/python scripts/render_spectral_science_math.py --check`，并检查公式图片和正文。
4. 将 Markdown、JSON、生成的 SVG 一起提交。不要手改 SVG，也不要在正文重新插入依赖插件的数学宏。

生成版本为 matplotlib 3.11.2。脚本禁用 SVG 时间戳并固定字形 ID 的哈希种子，同一版本可重复生成相同文件。
换 matplotlib／字体库版本后可能需要重新生成并复核；`--check` 的二进制差异不一定意味着数学内容变化。
全部渲染在本地完成，不使用远程公式图片服务，也不会把研究内容发给第三方渲染网站。
