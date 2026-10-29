# -*- coding: utf-8 -*-
"""界面渲染器与 app.py 的**接口契约**测试（不需要相机、不需要显示器）。

★ 为什么要有这个模块：`app.py --windowed` 长期没被任何测试覆盖，
  一跑就连崩两次，两次都是"跨模块接口不一致"：
    ① `app.py` 无条件传 `actions={...}` 给 `make_renderer()`，
       而 `OpenCVRenderer.__init__()` 没有这个参数
       -> TypeError: unexpected keyword argument 'actions'
    ② `OpenCVRenderer` 没调 `super().__init__()`，
       于是 `Renderer.run()` 里的 `self.frames_drawn` 不存在
       -> AttributeError: 'OpenCVRenderer' object has no attribute 'frames_drawn'
  QtRenderer 两条都没踩到（它是 QMainWindow 子类，链路里自然调到了基类），
  所以**只有 --windowed 的默认路径（opencv）是坏的**，而且没人发现。

★ 本模块的做法：**完全按 app.py 的真实参数**构造渲染器，只构造、不显示。
  这样接口一改就会在这里挂，而不是等到真机运行时。
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check, wp_config                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assemvision.ui import make_renderer                      # noqa: E402
from assemvision.ui.base import Renderer                      # noqa: E402
from assemvision.ui.base import UIState


def _app_kwargs(wp, W=1920, H=1080):
    """★ 与 app.py 里那次 make_renderer(...) 调用**逐字对应**。

    app.py 改了参数这里也要改 —— 这正是本测试的意义：
    两边不一致就会在离线回归里暴露，而不是等到真机。
    """
    return dict(
        window="装配流程检查",
        fullscreen=False,
        display=(W, H),
        panel_ratio=float(wp.get("ui.panel_ratio", 0.20)),
        font_path=str(wp.get("ui.font_path", "")),
        colors=wp.get("ui.colors", None),
        video_fps=int(wp.get("ui.video_fps", 30)),
        actions={"inspect": lambda: None, "reset": lambda: None},
    )


def main() -> int:
    print("== 界面渲染器接口契约（按 app.py 的真实参数构造）==")
    wp = wp_config()
    kw = _app_kwargs(wp)

    # ① OpenCV 渲染器（--windowed 的默认路径，曾经两处都坏）
    try:
        r = make_renderer("opencv", **kw)
        ok_open = True
    except Exception as e:                                     # noqa: BLE001
        ok_open = False
        print("   ★ opencv 渲染器构造失败: %s: %s" % (type(e).__name__, e))
    check(ok_open, "opencv 渲染器能吃下 app.py 的全部参数（含 actions）")

    if ok_open:
        # ② 必须继承 Renderer：`Renderer.run()` 依赖基类设的 frames_drawn
        check(isinstance(r, Renderer),
              "OpenCVRenderer 必须是 Renderer 子类（否则 run() 会缺属性）")
        check(hasattr(r, "frames_drawn"),
              "必须有 frames_drawn（Renderer.run 靠它做 --max-frames 计数）")
        check(int(r.frames_drawn) == 0, "frames_drawn 初值应为 0")
        # ③ actions 要真的被收下，且键名与 QtRenderer 一致
        check(set(r.actions.keys()) == {"inspect", "reset"},
              "actions 键应为 inspect / reset（与 QtRenderer 同名同义）")
        # ④ 抽象方法都被实现了，才能实例化/调用
        for name in ("render", "show", "close"):
            check(callable(getattr(r, name, None)),
                  "必须实现 %s()" % name)
        print("   中文字体: %s" % (r.font_path or "(没找到，会退回英文)"))

    # ⑤ Qt 渲染器：环境里没 PyQt5 就跳过，跳过要说清楚不是通过
    if ok_open:
        r.cache_unchanged = True
        state = UIState(frame=np.zeros((32,32,3),'uint8'),title='缓存原图')
        with patch.object(r,'_render_frame',wraps=r._render_frame) as draw:
            first = r.render(state)
            expected = first.copy()
            first[:] = 0
            assert np.array_equal(r.render(state), expected) and draw.call_count == 1
            state.title = '观测过期'
            r.render(state)
            assert draw.call_count == 2
            state.polygons = [(np.array([[1,1],[5,1],[5,5]],dtype='int32'),'ok',2)]
            r.render(state)
            assert draw.call_count == 3
            state.polygons[0][0][0,0] += 1
            r.render(state)
            assert draw.call_count == 4
            state.frame = state.frame.copy()
            r.render(state)
            assert draw.call_count == 5
        r.close()

    try:
        import PyQt5  # noqa: F401
        has_qt = True
    except ImportError:
        has_qt = False
    if has_qt:
        try:
            q = make_renderer("qt", **kw)
            check(isinstance(q, Renderer), "QtRenderer 也必须是 Renderer 子类")
            check(hasattr(q, "frames_drawn"), "QtRenderer 也要有 frames_drawn")
        except Exception as e:                                 # noqa: BLE001
            check(False, "qt 渲染器构造失败: %s" % e)
    else:
        print("   （环境里没有 PyQt5，跳过 qt 渲染器 —— 这里是**跳过不是通过**）")

    # ⑥ 未知渲染器名要明确报错，而不是悄悄退回
    try:
        make_renderer("nonsense", **kw)
        check(False, "未知渲染器名应当报错")
    except ValueError:
        check(True, "未知渲染器名会报 ValueError（不静默退回）")
    except Exception as e:                                     # noqa: BLE001
        check(False, "未知渲染器名抛的是 %s，应为 ValueError" % type(e).__name__)

    print("\n界面渲染器契约测试全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
