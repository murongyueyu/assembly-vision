# D任务卡：新A候选接入与稳定放行复验（2026-10-19）

仅D模块；实际基线2c10e4e5156e538c5c7293c8aa9b28082167a8d3，开工工作区干净，未push。原D任务卡及交付记录保留。

## 范围与具体文件

- 新增assembly_vision/config/prestart_check_wheel10_candidate.yaml：只替换identity的model、policy路径和两项hash，另存候选；其他D参数完全沿用，旧prestart_check.yaml和E绑定不变。
- 新增assembly_vision/tools/validate_prestart_candidate.py：D专用离线/现场验证工具，复用实际Pipeline与UI；保留逐帧原图、预测/身份确认/当前轨迹、C/D/allow_start、界面及SDK元数据；不改变算法。
- 新增assembly_vision/tests/test_prestart_candidate.py，登记tests/run_all.py：候选绑定校验、10px²特征传递、旧配置默认兼容及来源不符拒绝。
- 本任务卡、docs/D模块_新A候选稳定放行复验.md、docs/现状与交接.md、VERSION.md、docs/版本历史_详录.md、docs/对话存档_2026_D联检.md：准备和最终事实记录；有新坑再更新坑索引。
- ai_测试工具/D模块联检/20261019_01/：本机证据，受既有忽略规则保护，另行备份。

以上为用户本轮D配置/验证/文档授权。app.py、pipeline.py、core/part_identity.py暂不需要修改；若发现必要，先列具体位置、最小改动与理由并取得授权。A/B/C/E算法、阈值、模板及正式相机/几何/深度冻结，ui/types不改，不运行真实E工序。

## 已核验与兼容性

模型SHA256 c368d6d44ffda953ca63dea53688c8bd32c62837e60e2ef2d05f045c115d3c56。
policy SHA256 2af0b26d4969ced04bda10e1fcf670a6c31e276d83cf37465315c9a156e60266。
现有FrozenPartIdentity调用segment_variant先写入孔特征，extract_features消费已有特征；需实际回放证明候选10已生效，不以只加载成功代替兼容验证。候选内嵌policy与外部一致，正式配置hash须继续核验。

## 验证与放行规则

1. 指定D:/miniconda/envs/yolov11/python.exe；记录开工源码/配置hash，核验候选与旧v5来源。
2. 用既有A六组原图及原采集时刻回放真实A/C/D，分别报告原始分类、当前确认身份、D槽匹配及allow_start；旧失败仅开发回归。重点保存首组立柱尺寸拒识阻碍放行证据。
3. 新现场每次只提出一个动作并先等用户确认实际摆放：四件正确静置、整盘不同转角、互换错放、缺件、蓝手套遮挡、未知物体及恢复。原图目视复核真值，不按目录或提示标注。
4. 正确组须稳定放行，错误/缺件/未知/遮挡组不得放行；不改尺寸必检、七帧0.85或槽位身份先验。失败留证交A另一个对话，不绕过。
5. 全量测试（当前25模块，新增后26）、真实Pipeline离线端到端及实际--windowed；冻结hash/工作区审计。现场未完成不能宣称稳定通过，不建议进入E。
6. 完成文档、AGENTS≤20KB、VERSION≤8KB及本地提交，不push。验证通过前不覆盖正式绑定。

## 状态

已建立任务卡与另存候选配置；兼容性及验证进行中，现场尚未开始，D稳定放行未验收。


## rev.3w.2最终执行状态

D孔候选接入无需配套生产代码修改；现场164次53放行，稳定检查未通过。用户停止现场并连续三次确认跨A例外，随后另存尺寸上限开发候选及D消费配置，详见追加A任务卡。本轮实际D/A，不再声称仅D；27模块与开发回归通过，未现场复测，正式D/E不替换，E/G未推进，详见docs/D模块_新A候选稳定放行复验.md。
