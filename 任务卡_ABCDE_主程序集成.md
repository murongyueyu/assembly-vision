# ABCDE主程序集成任务卡（2026-10-19）

## 授权与开工核对
用户明确允许本轮ABCDE必要跨模块接线，作为单模块纪律专项例外；不转G，不push。实际HEAD=65cb3326f502920316813f8f26fd9ab617ead15a，开工工作区干净。
尺寸候选model SHA256=214991c9f31d99214d42466b94125242a56c998f599addea92ae66711f11e78a；policy=2af0b26d4969ced04bda10e1fcf670a6c31e276d83cf37465315c9a156e60266，实核一致。
已读AGENTS、现状/架构、A/B/C/D/E当前任务卡、D rev.3w.3、E rev.3x.1.1和B原生深度交付。B只有人工选顶盖表面的受控验证，baseline_accepted=false；禁止旧固定像素区自动关联。

## 具体文件与最小改动
- assembly_vision/app.py：显式--integration入口，绑定集成候选；复用现有窗口及i/R动作。
- assembly_vision/assemvision/pipeline.py：复用现有A/C/D/E同帧分支；集成专用双流采集和B快照测量，禁止启动旧流程/旧深度。
- assembly_vision/assemvision/integration.py：只做运行配置/hash绑定、B受控同帧关联与证据/UI编排，不改core算法。
- assembly_vision/config/abcde_integration_candidate.yaml：另存明确A/C/D/E候选及B冻结运行副本绑定；正式默认配置不变。
- assembly_vision/tools/validate_abcde_integration.py：单一真实app证据工具，运行中保存检查点、原图/UI/事件/测区/深度/相机元数据，外部控制不改变算法。
- assembly_vision/tests/test_abcde_integration.py及tests/run_all.py：同帧、门禁、四步、过期/复位、B关联/基线拒绝及入口兼容测试。
- 本卡、docs/ABCDE_主程序集成.md、现状§1/2/3、架构命令、VERSION、版本详录、对话追加存档；发现新坑再写实录及索引。

## 冻结与B边界
A/B/C/D/E core算法、阈值、模型、policy、标定、基线冻结。B沿用原生SDK及质量门禁；不得用旧测区猜当前装配体。当前不存在已验收自动表面关联，因此默认待测。E四步后，允许操作者在明确帧号的冻结彩色/深度同帧上选裸绿垫参考和当前装配体顶盖顶部小片，作为一次人工关联的受控测量；取消/不关联不测。不得将快照结论冒充持续自动定位。baseline_accepted保持false，数值测量成功仍显示高度符合性无法判断。安装/紧固等未验证。

## 验收条件与顺序
1. 已有材料离线回归，真实Pipeline同帧A/C/D/E接线，D通过才启动E，启动后空槽由E解释；4/4仅取件完成。
2. 集成候选配置明确校验来源；全量回归（现29模块+新增）、旧合成端到端及--windowed兼容、新集成入口实际--windowed。
3. 一轮真实正常顺序，每次仅一个现场动作、用户确认后记录对应采集；窗口持续运行，失败先留证不放宽门槛。
4. 真实B有可靠人工表面关联才测；否则待测/无法判断；分别报告离线、真实取件、测量、符合性与未测质量项。
5. 运行中证据落盘；核对原图、事件、测区、深度、元数据；文档体量及本地提交，不push。

当前：准备完成，实施中；未开始本轮现场采集。

## 用户追加授权与交付状态

现场后用户明确要求测量在正常范围应判合格，否决原±15mm并指定容差最多3mm；紧固性由队友完成、本程序暂不实现。最小改动为integration.py复用既有纯比较函数，另存abcde_integration_height_3mm_candidate.yaml，仅接受人工同帧顶盖高度基线68.025114mm及±3mm；旧B配置/基线数值、core算法和其他阈值不变。测试补充上下超差/最大容差拒绝，保存同一真实快照授权后重评，不重复现场采集。

用户随后明确本轮需推送GitHub，并要求队友交接文档，覆盖初始不push约定。新增docs/ABCDE_队友交接.md、docs/ABCDE_运行依赖清单.json和本机忽略目录内运行依赖ZIP；写清分工、启动、紧固性结果交回协议与尚未实现的回传接口。

已完成：30模块、旧合成4/4/窗口、新集成及3mm候选窗口；真实首次3/4，顶盖单步补充后4/4；人工同帧测量67.654270mm，按用户±3mm授权重评高度合格。2628文件hash、时序/控件/冻结审计、exit0及线程/相机释放通过。现场不补称完整一遍无补充或全面质量通过，紧固性归队友，G未推进。文档、本地提交及GitHub推送按最终授权收尾。
