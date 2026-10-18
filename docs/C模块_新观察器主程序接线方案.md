# 新C观察器接入主程序：独立C观察模式

当前为rev.3t.5：用户已明确授权并实施下述五处配套，主程序独立C观察模式接入通过。入口为`app.py --tray-observer config/tray_observation_v2.yaml --windowed`，期限配置由`--tray-observer-runtime`传入，默认`config/tray_observer_runtime.yaml`。默认旧模式兼容；新C未被生产D/E消费，真实完整流程未验收。20模块、主入口105帧与直接工具逐帧一致、真实D435四次窗口空/满/三件/转盘检查通过；拒识/过期/异常清旧及退出释放已验证。[完整实施及限制](C模块_空满料盘识别.md#rev3t5主程序独立c观察模式接线与验证2026-10-18)，原始证据在20261018_04。下文「待授权/拟新增/建议」均为上一轮方案原文，仅保留设计依据，不表示仍未实施。

## 原最小方案（实施前记录）

2026-10-18，基于9a0a928主链代码核对，并纳入本轮独立v2修正候选接口。本文件是具体实施提案，尚未修改app.py/pipeline.py。原C几何两处例外授权不适用于本提案。v2修正的现场验证状态见C观察器记录，不以接线文档替代验收。

## 第一步的行为与文件范围

建议先在同一个app.py入口增加显式的「C观察模式」，运行新C的定位和占用显示；不把占用状态写进旧SlotResult.present/verdict、FSM或事件。这样可先验证主程序入口下的同帧采集、加载与显示，不必同时改D/E语义或装配定位。此模式的通过仅称「主程序C观察模式接入通过」，不称「生产主链替换通过」或「完整流程通过」。

待授权跨文件生产改动只有以下两处；另新增C专用显示运行配置（见下文），不改候选参数：

| 文件 | 具体改动 |
|---|---|
| assembly_vision/app.py | 新增可选--tray-observer配置路径（默认不开启）；按现有_p规则解析后传Pipeline；C模式禁用inspect/reset动作，避免触发旧装配抽检；帮助与日志明确为C占用观察 |
| assembly_vision/assemvision/pipeline.py | 可选构造参数、独立C初始化分支、同帧处理/发布和UIState组装；C模式仅启动现有采集线程与C处理线程，不启动旧逻辑/位姿/深度/抽检线程；旧模式保持原入口 |

配套测试增加assembly_vision/tests/test_tray_observer_integration.py，并登记tests/run_all.py；需一并明确授权这两个测试文件。C算法、参数、参考PNG、相机控件及正式配置均不改；不改types.py、checks.py、fsm.py、events.py或ui/*。UIState已有frame/detail/polygons/stats可承载结果，无需扩展公共判据数据类。

## 调用关系及帧一致性

```text
app --tray-observer config/tray_observation.yaml
  -> Pipeline(..., tray_observer_path=已解析绝对路径)
  -> C模式初始化TrayObserver.from_file(wp,cal,path)及独立Segmenter(wp)
  -> source.open / 当前控件锁定 / 预热（沿用D435来源工厂）
  -> T1发布Frame
  -> C处理线程取得一份Frame（不在处理过程中再取最新帧）
     -> 验证left为640×480、scale=1，不用插值伪造分辨率
     -> Segmenter.segment(frame.left, ts=frame.ts, scale=1,
                          mm_per_px=observer.tray.mm_per_px())
        与独立工具一致，不传旧_work_roi或缓存位姿
     -> observe(frame.left, seg.metal_mask, seg.glove_mask,
                brass_mask=seg.brass_mask)
     -> 锁内一次发布(frame.index, frame.ts, frame.left, observation)
  -> build_ui_state取同一份快照并在对应原图画C槽位
```

每个frame.index至多处理一次，跳过重复轮询；不累积大图队列。锁外计算、锁内交换整份快照。UI不能用「最新采集帧」叠加「上一帧观测」。处理线程异常要显式发布失败并通知退出，不沿用最后成功。加载时沿用几何绑定/参考hash校验，失败直接报错；显式开启失败不得静默退回旧轮廓估计。

C模式不调用旧_seg_loop的_work_roi链，不调用Qualifier、PoseChecker及_last_pose回退；旧手套命中、身份标签、装配ROI和齐套性文字不进入C界面。C角度只表示当前自动估计，不冒充外部角度真值。

模型路径必须显式选择已核验的候选。原tray_observation.yaml保持metal占用模式；v2增加matching.exclude_brass及occupancy.evidence=metal_or_brass，分别控制匹配排除和占用双证据。主程序不能擅自把黄铜掩膜冒充金属掩膜传入，也不能把旧配置加载成功当新候选启用；沿用C返回的metal_fraction/brass_fraction/occupancy_fraction/evidence记录核查。

## 输出与拒识

每槽显示空/有件/不确定，整体显示空盘/满盘/部分占用/不确定；标题明确「C料盘占用观察」，详情保留「有件不代表身份正确或合格」。orientation_ok=false时四槽全部显示不确定并移除槽位身份叠加，不拿slots空字典当四槽空。

等待首帧或采集停滞时显示等待/观测过期，不显示当前确认。拟新增C专用config/tray_observer_runtime.yaml，定义display.max_age_s: 1.0，作为可审查的初始显示期限（并非已实测验收的响应保证）；与约75ms的既有算法P95分开记录。使用本机time.monotonic记录采集/发布时刻，不拿文件回放ts直接和墙钟比较；有效性依据采集时刻，不能只用发布时刻让积压旧帧变新。阈值不得硬编码进算法。该配置只影响显示新鲜度，不修改冻结tray_observation.yaml的匹配/占用参数或绑定。

## 必要验证及边界

1. 配套行为测试覆盖启动开关、错误模型/几何拒绝、掩膜与原图同帧、重复帧不重复处理、方向拒识后清除旧四槽、采集过期及线程异常可见；不以UI轮询次数代替相机帧数。
2. 已人工复核的新独立数据经主入口回放，与独立C工具逐帧结果对齐；输入沿用同一工位正式calibration，不用不绑定的synthetic calibration加载现场参考。
3. D435现场--windowed：空/满/部分占用及转盘，核对原图与框；关闭后相机/线程释放。19模块及新增模块通过，再做旧模式--windowed兼容冒烟。
4. 旧合成端到端按实际结果记录；目前已知2/4，不为通过而改D/E。本方案不向D提供新pose，也不替换旧装配ROI或槽内零件合格性链。

## 后续生产主链替换为何另立授权

旧pipeline._logic_loop分别读取共享seg和最新frame，存在跨帧风险；旧check_slots即使收到pose仍调用pose_estimate提取识别，旧work_roi/assembly_flags又会将装配区域随盘移动。不能只给check_slots塞一个新pose就声称完整接通。第二步若要求生产流程消费新C，需另提同帧协议、当前未知门禁、独立装配定位及D/E接口方案；不在本轮最小C接线授权内。
