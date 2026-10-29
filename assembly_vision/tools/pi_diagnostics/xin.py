#!/usr/bin/env python3
"""极简 X 输入注入（ctypes + libXtst），用于远程操作树莓派上的 cv2 窗口。

用法：
  xin.py windows                         列出顶层窗口的名字与几何
  xin.py click X Y                       左键单击
  xin.py drag X1 Y1 X2 Y2                左键拖拽（用于 cv2.selectROI）
  xin.py key i                           发一个按键（keysym 名字，如 i / Return / Escape）
  xin.py seq click 500 300 key i         连做多个动作
"""
import ctypes
import ctypes.util
import sys
import time

x11 = ctypes.CDLL(ctypes.util.find_library('X11'))
xtst = ctypes.CDLL(ctypes.util.find_library('Xtst'))

x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XDefaultRootWindow.restype = ctypes.c_ulong
x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x11.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                           ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)),
                           ctypes.POINTER(ctypes.c_uint)]
x11.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_char_p)]
x11.XGetGeometry.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
                             ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
x11.XTranslateCoordinates.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                                      ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_ulong)]
x11.XStringToKeysym.restype = ctypes.c_ulong
x11.XStringToKeysym.argtypes = [ctypes.c_char_p]
x11.XKeysymToKeycode.restype = ctypes.c_ubyte
x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
x11.XFlush.argtypes = [ctypes.c_void_p]
xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_ulong]
xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
xtst.XTestFakeKeyEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

D = x11.XOpenDisplay(None)
if not D:
    sys.exit('cannot open display (DISPLAY/XAUTHORITY?)')
ROOT = x11.XDefaultRootWindow(D)


def windows():
    root_ret, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children = ctypes.POINTER(ctypes.c_ulong)()
    n = ctypes.c_uint()
    x11.XQueryTree(D, ROOT, ctypes.byref(root_ret), ctypes.byref(parent),
                   ctypes.byref(children), ctypes.byref(n))
    out = []
    for i in range(n.value):
        win = children[i]
        name = ctypes.c_char_p()
        label = ''
        if x11.XFetchName(D, win, ctypes.byref(name)) and name.value:
            label = name.value.decode('utf-8', 'replace')
        x, y = ctypes.c_int(), ctypes.c_int()
        w, h = ctypes.c_uint(), ctypes.c_uint()
        bw, depth = ctypes.c_uint(), ctypes.c_uint()
        r = ctypes.c_ulong()
        if not x11.XGetGeometry(D, win, ctypes.byref(r), ctypes.byref(x), ctypes.byref(y),
                                ctypes.byref(w), ctypes.byref(h), ctypes.byref(bw), ctypes.byref(depth)):
            continue
        ax, ay = ctypes.c_int(), ctypes.c_int()
        x11.XTranslateCoordinates(D, win, ROOT, 0, 0, ctypes.byref(ax), ctypes.byref(ay), ctypes.byref(r))
        if w.value == 0 or h.value == 0:
            continue
        out.append((label, ax.value, ay.value, w.value, h.value))
    return out


def move(x, y):
    xtst.XTestFakeMotionEvent(D, -1, int(x), int(y), 0)
    x11.XFlush(D)
    time.sleep(0.05)


def click(x, y, button=1):
    move(x, y)
    xtst.XTestFakeButtonEvent(D, button, 1, 0)
    x11.XFlush(D)
    time.sleep(0.05)
    xtst.XTestFakeButtonEvent(D, button, 0, 0)
    x11.XFlush(D)
    time.sleep(0.15)


def drag(x1, y1, x2, y2, steps=12):
    move(x1, y1)
    xtst.XTestFakeButtonEvent(D, 1, 1, 0)
    x11.XFlush(D)
    time.sleep(0.08)
    for i in range(1, steps + 1):
        move(x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
    time.sleep(0.08)
    xtst.XTestFakeButtonEvent(D, 1, 0, 0)
    x11.XFlush(D)
    time.sleep(0.15)


def key(name):
    sym = x11.XStringToKeysym(name.encode())
    if not sym:
        sys.exit('unknown keysym: %s' % name)
    code = x11.XKeysymToKeycode(D, sym)
    xtst.XTestFakeKeyEvent(D, code, 1, 0)
    x11.XFlush(D)
    time.sleep(0.05)
    xtst.XTestFakeKeyEvent(D, code, 0, 0)
    x11.XFlush(D)
    time.sleep(0.15)


def find(substr):
    for label, x, y, w, h in windows():
        if substr in label:
            return label, x, y, w, h
    sys.exit('no window matching %r' % substr)


def movewin(substr, x, y):
    label, ox, oy, w, h = find(substr)
    root_ret, parent = ctypes.c_ulong(), ctypes.c_ulong()
    children = ctypes.POINTER(ctypes.c_ulong)()
    n = ctypes.c_uint()
    x11.XQueryTree(D, ROOT, ctypes.byref(root_ret), ctypes.byref(parent),
                   ctypes.byref(children), ctypes.byref(n))
    for i in range(n.value):
        win = children[i]
        name = ctypes.c_char_p()
        ok = x11.XFetchName(D, win, ctypes.byref(name))
        if ok and name.value and substr.encode() in name.value:
            x11.XMoveWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
            x11.XMoveWindow(D, win, int(x), int(y))
            x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            x11.XRaiseWindow(D, win)
            x11.XSync(D, 0)
            print('moved %r -> %d,%d (was %d,%d %dx%d)' % (label, x, y, ox, oy, w, h))
            return 0
    sys.exit('window not found for move')


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cmd = argv[0]
    if cmd == 'geom':
        label, x, y, w, h = find(argv[1])
        print('%s x=%d y=%d w=%d h=%d' % (repr(label), x, y, w, h))
        return 0
    if cmd == 'movewin':
        return movewin(argv[1], float(argv[2]), float(argv[3]))
    if cmd == 'windows':
        for label, x, y, w, h in windows():
            print('%-58s x=%4d y=%4d w=%4d h=%4d' % (repr(label), x, y, w, h))
        return 0
    if cmd == 'seq':
        rest = argv[1:]
        while rest:
            name, rest = rest[0], rest[1:]
            if name == 'click':
                click(float(rest[0]), float(rest[1])); rest = rest[2:]
            elif name == 'drag':
                drag(*[float(v) for v in rest[:4]]); rest = rest[4:]
            elif name == 'key':
                key(rest[0]); rest = rest[1:]
            elif name == 'sleep':
                time.sleep(float(rest[0])); rest = rest[1:]
            else:
                sys.exit('bad action: %s' % name)
        x11.XSync(D, 0)
        return 0
    if cmd == 'click':
        click(float(argv[1]), float(argv[2])); x11.XSync(D, 0); return 0
    if cmd == 'drag':
        drag(*[float(v) for v in argv[1:5]]); x11.XSync(D, 0); return 0
    if cmd == 'key':
        key(argv[1]); x11.XSync(D, 0); return 0
    sys.exit(__doc__)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
