# -*- coding: utf-8 -*-
"""C模块：空盘槽形模板解完整朝向，输出槽内占用观测，不判工序或合格性。

金属/手套掩膜由冻结的A分割提供，本模块不修改或拟合A算法。
无方向证据、低分或四分支近似等价时输出unknown，不沿用历史身份。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy
import multiprocessing
from multiprocessing import shared_memory
import threading
from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import Config, load_config
from ..imgio import imread_safe
from .roi import TrayMap, capsule_polygon, circle_polygon
from .tray_detect import detect_tray, pose_from_tray_detect
from .tray_pose import TrayPose


@dataclass
class TrayObservation:
    found: bool = False
    orientation_ok: bool = False
    state: str = 'unknown'
    pose: Optional[TrayPose] = None
    slots: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    score: float = -1.
    margin: float = 0.
    reason: str = ''
    # ★ 这是单帧直接观测，不把缓存位姿当当前实测。
    observed_current: bool = True

    def to_dict(self):
        return asdict(self)


def geometry_binding(wp: Config, cal: Config) -> dict:
    """只绑定C几何；A/B的模型和阈值不纳入本模块重拟合。"""
    return {'homography': cal.get('tray_map.homography'),
            'ref_scale': float(cal.get('tray_map.scale', 1.)),
            'tray_size_mm': wp.get('tray.size'),
            'slots': [{key: slot.get(key) for key in
                       ('id', 'type', 'center', 'diameter', 'length', 'width', 'angle_deg')}
                      for slot in wp.get('slots')]}


def shape_feature(gray, mode):
    """梯度幅值削弱凹槽两侧明暗随旋转互换的影响，不利用零件身份定向。"""
    gray = np.asarray(gray, 'float32')
    if mode == 'gradient':
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                             cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    if mode == 'gray':
        return gray
    raise ValueError('未知C槽形特征：'+str(mode))


class _CorrelationCache:
    """仅缓存由当前模板/当前遮挡掩膜共同决定的数学项，不缓存位姿或判定。"""
    def __init__(self, capacity=64):
        self.data = OrderedDict()
        self.lock = threading.Lock()
        self.capacity = capacity
        self.hits = 0

    def get(self,key):
        with self.lock:
            value=self.data.get(key)
            if value is not None:
                self.data.move_to_end(key)
                self.hits+=1
            return value

    def put(self,key,value):
        with self.lock:
            self.data[key]=value
            self.data.move_to_end(key)
            while len(self.data)>self.capacity:
                self.data.popitem(last=False)

    def clear(self):
        with self.lock:
            self.data.clear()


def masked_correlation(reference, current, valid, border, search, min_visible, min_std,
                       backend='opencv',cache=None):
    """一次求出平移窗口的加权NCC；遮挡随当前图移动，不能补零冒充模板证据。"""
    h, w = reference.shape
    target = np.asarray(reference[border:h-border, border:w-border], 'float32')
    image = np.asarray(current[border-search:h-border+search,
                               border-search:w-border+search], 'float32')
    weight = valid[border-search:h-border+search,
                   border-search:w-border+search].astype('float32')
    ones = np.ones_like(target)
    corr = lambda src, tpl: cv2.matchTemplate(src, tpl, cv2.TM_CCORR)
    if backend in ('box_sum','box_sum_cached','fft_cached'):
        # ★ 与全1模板相关等价的窗口求和；仍搜索原来的每一个平移位置。
        th, tw = target.shape
        def total(src):
            sums = cv2.boxFilter(src, cv2.CV_32F, (tw, th), anchor=(0,0),
                                 normalize=False, borderType=cv2.BORDER_CONSTANT)
            return sums[:image.shape[0]-th+1, :image.shape[1]-tw+1]
    elif backend == 'opencv':
        total = lambda src: corr(src, ones)
    else:
        raise ValueError('未知C相关计算后端：'+str(backend))
    if backend == 'fft_cached':
        # ★ 原加权相关的频域计算；模板频谱固定，当前图/遮挡仍逐帧变换。
        dh,dw = (cv2.getOptimalDFTSize(n) for n in image.shape)
        sources = {}
        def spectrum(src):
            padded = cv2.copyMakeBorder(src,0,dh-src.shape[0],0,dw-src.shape[1],
                                       cv2.BORDER_CONSTANT,value=0)
            return cv2.dft(padded,flags=cv2.DFT_COMPLEX_OUTPUT)
        def corr(src,tpl):
            item = sources.get(id(src))
            if item is None:
                item=(src,spectrum(src))
                sources[id(src)] = item
            template_key=('fft',tpl.shape,tpl.tobytes(),dh,dw)
            template = cache.get(template_key) if cache is not None else None
            if template is None:
                template=spectrum(tpl)
                if cache is not None:
                    cache.put(template_key,template)
            product=cv2.mulSpectrums(item[1],template,flags=0,conjB=True)
            result=cv2.idft(product,flags=cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
            return result[:image.shape[0]-tpl.shape[0]+1,:image.shape[1]-tpl.shape[1]+1].copy()
        # 频谱缓存不保存当前遮挡项，避免真实相机掩膜微动造成无效缓存。
        value_cache=None
    else:
        value_cache=cache
    key=(target.shape,target.tobytes(),weight.shape,weight.tobytes()) if value_cache is not None else None
    values=value_cache.get(key) if value_cache is not None else None
    if values is None:
        count=total(weight)
        sum_a=corr(weight,target)
        sum_a2=corr(weight,target*target)
        if value_cache is not None:
            value_cache.put(key,(count,sum_a,sum_a2))
    else:
        count,sum_a,sum_a2=values
    safe_count = np.maximum(count, 1.)
    sum_b = total(weight * image)
    sum_b2 = total(weight * image * image)
    cross = corr(weight * image, target)
    var_a = np.maximum(sum_a2 - sum_a * sum_a / safe_count, 0.)
    var_b = np.maximum(sum_b2 - sum_b * sum_b / safe_count, 0.)
    denominator = np.sqrt(var_a * var_b)
    scores = np.full_like(count, -1.)
    supported = ((count >= target.size * min_visible)
                 & (var_a >= safe_count * min_std ** 2)
                 & (var_b >= safe_count * min_std ** 2))
    scores[supported] = ((cross - sum_a * sum_b / safe_count)[supported]
                          / denominator[supported])
    scores = np.clip(scores, -1., 1.)
    y, x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return float(scores[y, x]), (int(x-search), int(y-search)), float(count[y, x]/target.size)


_PROCESS_OBSERVER = _PROCESS_MEMORY = _PROCESS_PLANES = None


def _init_direction_process(wp, cal, settings, reference, views, name, shape, backend):
    """spawn子进程只构造C，不打开相机；共享内存里的当前帧只读。"""
    global _PROCESS_OBSERVER, _PROCESS_MEMORY, _PROCESS_PLANES
    cv2.setNumThreads(1)
    _PROCESS_OBSERVER = TrayObserver(Config(wp),Config(cal),settings,reference,
                                     correlation_backend=backend)
    _PROCESS_OBSERVER.reference_views = views
    _PROCESS_MEMORY = shared_memory.SharedMemory(name=name)
    _PROCESS_PLANES = np.ndarray(shape,dtype=np.uint8,buffer=_PROCESS_MEMORY.buf)
    _PROCESS_PLANES.flags.writeable = False


def _direction_process(job):
    serial, branch, angle, center = job
    result = _PROCESS_OBSERVER._evaluate_branch(branch,*_PROCESS_PLANES,angle,np.asarray(center))
    return serial,result


class TrayObserver:
    """模板与配置均为显式输入；未知掩膜、失配标定不静默退回固定槽位。"""

    def __init__(self, wp: Config, cal: Config, settings: dict, reference: np.ndarray,
                 workers=1, correlation_backend='opencv', execution_backend='thread'):
        if type(workers) is not int or not 1 <= workers <= 4:
            raise ValueError('C方向并行数必须为1～4的整数')
        if correlation_backend not in ('opencv', 'box_sum','box_sum_cached','fft_cached'):
            raise ValueError('C相关计算后端无效')
        self.correlation_backend = correlation_backend
        self._correlation_cache = _CorrelationCache() if correlation_backend in ('box_sum_cached','fft_cached') else None
        if execution_backend not in ('thread','process'):
            raise ValueError('C方向执行后端无效')
        self.execution_backend = execution_backend
        self._cal_data = copy.deepcopy(cal.data)
        if settings['geometry_binding'] != geometry_binding(wp, cal):
            raise ValueError('C槽形模板与当前c3/槽位几何不一致，需要重新建立C模板')
        self.wp, self.settings = wp, settings
        self.tray = TrayMap(wp, homography=np.asarray(cal.get('tray_map.homography'), float),
                            ref_scale=float(cal.get('tray_map.scale', 1.)))
        self.ppm = float(settings['canonical_pixels_per_mm'])
        self.size = np.asarray(wp.get('tray.size'), float)
        self.canvas = tuple(np.rint(self.size * self.ppm).astype(int))
        if reference is None or reference.shape != (self.canvas[1], self.canvas[0]):
            raise ValueError('C空盘模板尺寸无效')
        self.reference = reference.astype('float32')
        self.feature_mode = settings['matching'].get('feature', 'gray')
        self.reference_feature = shape_feature(self.reference, self.feature_mode)
        self.reference_views = []
        self.slot_masks = {}
        inset = float(settings['occupancy']['inset_mm'])
        for sid, geometry in settings['rim_geometry'].items():
            if 'rim_diameter_mm' in geometry:
                poly = circle_polygon(geometry['center_mm'], geometry['rim_diameter_mm']-2*inset)
            else:
                poly = capsule_polygon(geometry['center_mm'], geometry['rim_length_mm']-2*inset,
                                       geometry['rim_width_mm']-2*inset, geometry['angle_deg'])
            mask = np.zeros(self.reference.shape, 'uint8')
            cv2.fillPoly(mask, [np.rint(poly*self.ppm).astype('int32')], 255)
            self.slot_masks[sid] = mask > 0
        if set(self.slot_masks) != {s['id'] for s in self.tray.slots}:
            raise ValueError('C模板槽位集合不完整')
        self.workers = workers
        self._closed = False
        self._process_lock = threading.Lock()
        self._shared_memory = self._shared_planes = None
        self._frame_serial = 0
        self._executor = (ThreadPoolExecutor(max_workers=workers, thread_name_prefix='C-angle')
                          if workers > 1 and execution_backend == 'thread' else None)

    def close(self):
        """调用方停止处理线程后释放方向线程池，禁止关闭后继续发布结果。"""
        with self._process_lock:
            self._closed = True
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None
            if self._shared_memory is not None:
                self._shared_planes = None
                self._shared_memory.close()
                self._shared_memory.unlink()
                self._shared_memory = None
            if self._correlation_cache is not None:
                self._correlation_cache.clear()

    @classmethod
    def from_file(cls, wp, cal, path, workers=1, correlation_backend='opencv',
                  execution_backend='thread'):
        path = Path(path)
        settings = load_config(str(path)).data
        reference_path = path.parent / settings['reference_image']
        if hashlib.sha256(reference_path.read_bytes()).hexdigest() != settings['reference_sha256']:
            raise ValueError('C空盘模板文件hash不匹配')
        image = imread_safe(str(reference_path))
        if image is None:
            raise ValueError('读不到C空盘模板')
        reference = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        observer = cls(wp, cal, settings, reference, workers=workers,
                       correlation_backend=correlation_backend, execution_backend=execution_backend)
        for view in settings.get('reference_views', []):
            view_path = path.parent / view['image']
            if hashlib.sha256(view_path.read_bytes()).hexdigest() != view['sha256']:
                raise ValueError('C方向参考文件hash不匹配')
            image = imread_safe(str(view_path))
            if image is None or image.shape[:2] != reference.shape:
                raise ValueError('C方向参考尺寸无效')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            observer.reference_views.append((float(view['theta_deg']), shape_feature(gray, observer.feature_mode)))
        return observer

    def reference_for_angle(self, angle):
        """固定光源下槽边随转动改变明暗，选择最近实测方向的空盘参考。"""
        if not self.reference_views:
            return self.reference_feature
        return min(self.reference_views,
                   key=lambda item: abs((angle-item[0]+180.)%360.-180.))[1]

    def image_homography(self, pose):
        transform = np.eye(3)
        transform[:2, :2] = pose.R
        transform[:2, 2] = (pose.tx, pose.ty)
        # ★ H保存在ref_scale；当前入口只接受绑定的原始分辨率。
        image_h = self.tray.H.copy()
        image_h[:2] /= self.tray.ref_scale
        return image_h @ transform

    def _warp(self, image, pose, nearest=False):
        matrix = np.diag([self.ppm, self.ppm, 1.]) @ np.linalg.inv(self.image_homography(pose))
        return cv2.warpPerspective(image, matrix, self.canvas,
                                   flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

    def _evaluate_branch(self, branch, gray, blocked, valid_image, base_angle, observed_center):
        match = self.settings['matching']
        branch_candidates = []
        for angle_offset in match.get('angle_offsets_deg', [0.]):
            pose = TrayPose(theta_deg=base_angle+90*branch+float(angle_offset),ok=True)
            pose.tx,pose.ty = map(float,observed_center-pose.R@(self.size/2))
            canonical = shape_feature(self._warp(gray,pose),self.feature_mode)
            visible = (self._warp(blocked,pose,True)==0)&(self._warp(valid_image,pose,True)>0)
            score,offset,support = masked_correlation(
                self.reference_for_angle(pose.theta_deg),canonical,visible,
                int(round(match['border_mm']*self.ppm)),int(round(match['translation_search_mm']*self.ppm)),
                float(match['min_visible_fraction']),float(match['min_std']),backend=self.correlation_backend,
                cache=self._correlation_cache)
            delta=pose.R@(np.asarray(offset)/self.ppm)
            pose.tx+=float(delta[0]);pose.ty+=float(delta[1])
            branch_candidates.append((score,support,branch,offset,pose))
        return max(branch_candidates,key=lambda item:item[0])

    def _process_candidates(self, gray, blocked, valid_image, base_angle, center):
        # ★ 一组方向全部完成前不覆盖共享帧；每个结果携带组号，绝不拼接不同帧。
        with self._process_lock:
            if self._closed:
                raise RuntimeError('C方向观察器已关闭')
            if self._executor is None:
                shape=(3,*gray.shape)
                self._shared_memory=shared_memory.SharedMemory(create=True,size=int(np.prod(shape)))
                self._shared_planes=np.ndarray(shape,dtype=np.uint8,buffer=self._shared_memory.buf)
                self._executor=ProcessPoolExecutor(max_workers=self.workers,
                    mp_context=multiprocessing.get_context('spawn'),initializer=_init_direction_process,
                    initargs=(self.wp.data,self._cal_data,self.settings,self.reference,self.reference_views,
                              self._shared_memory.name,shape,self.correlation_backend))
            self._shared_planes[0]=gray
            self._shared_planes[1]=blocked
            self._shared_planes[2]=valid_image
            self._frame_serial+=1
            serial=self._frame_serial
            jobs=[(serial,branch,base_angle,center.tolist()) for branch in range(4)]
            results=list(self._executor.map(_direction_process,jobs))
            if any(number!=serial for number,_ in results):
                raise RuntimeError('C方向进程返回了不同帧组')
            return [result for _,result in results]

    def observe(self, bgr, metal_mask, glove_mask=None, brass_mask=None):
        if self._closed:
            raise RuntimeError('C方向观察器已关闭')
        result = TrayObservation()
        expected = tuple(self.settings['image_size'])
        if bgr is None or (bgr.shape[1], bgr.shape[0]) != expected:
            result.reason = '图像分辨率与C模板不一致'
            return result
        for mask in (metal_mask, glove_mask, brass_mask):
            if mask is not None and mask.shape != bgr.shape[:2]:
                raise ValueError('C输入掩膜必须与当前原图同尺寸')
        if metal_mask is None:
            result.reason = '缺少当前帧金属观测'
            return result
        exclude_brass = bool(self.settings['matching'].get('exclude_brass', False))
        evidence = self.settings['occupancy'].get('evidence', 'metal')
        if evidence not in ('metal', 'metal_or_brass'):
            raise ValueError('未知C占用证据模式：'+str(evidence))
        if (exclude_brass or evidence == 'metal_or_brass') and brass_mask is None:
            result.reason = '缺少当前帧黄铜观测掩膜'
            return result
        detect = self.settings['outline']
        det = detect_tray(bgr, **detect)
        result.found = det.ok
        if not det.ok:
            result.reason = det.reason
            return result
        center_px = self.tray.mm_to_px([self.size/2])[0]
        base = pose_from_tray_detect(det, center_px, 1., self.tray.mm_per_px(),
                                     nominal_angle_deg=float(self.wp.get('tray_detect.nominal_angle_deg')),
                                     pixel_to_mm=self.tray.px_to_mm)
        observed_center = self.tray.px_to_mm([det.center_px])[0]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        match = self.settings['matching']
        gray = cv2.GaussianBlur(gray, (int(match['blur_kernel']),)*2, float(match['blur_sigma']))
        glove = np.zeros_like(metal_mask) if glove_mask is None else glove_mask
        blocked = cv2.bitwise_or(metal_mask, glove)
        if exclude_brass:
            # ★ 黄铜反光纹理不能当槽形；附加掩膜只用于匹配，不改下方占用输入。
            blocked = cv2.bitwise_or(blocked, brass_mask)
        blocked = cv2.dilate(blocked, np.ones((int(match['occlusion_dilate_px']),)*2, 'uint8'))
        valid_image = np.full(bgr.shape[:2], 255, 'uint8')
        # ★ 四分支采用同样的小角搜索，不能只细化赢家从而虚增方向差距。
        def evaluate_branch(branch):
            return self._evaluate_branch(branch,gray,blocked,valid_image,base.theta_deg,observed_center)
        if self.execution_backend=='process' and self.workers>1 and all(
                image.dtype==np.uint8 for image in (gray,blocked,valid_image)):
            candidates=self._process_candidates(gray,blocked,valid_image,base.theta_deg,observed_center)
        elif self._executor is None or self.execution_backend=='process':
            candidates = [evaluate_branch(branch) for branch in range(4)]
        else:
            # map保留原分支顺序，等分时仍采用原串行顺序，不按任务完成先后选赢家。
            candidates = list(self._executor.map(evaluate_branch, range(4)))
        candidates.sort(key=lambda item: -item[0])
        result.candidates = [{'branch': b, 'score': s, 'visible_fraction': v,
                              'translation_px': list(o), 'theta_deg': p.theta_deg}
                             for s,v,b,o,p in candidates]
        score, support, branch, offset, pose = candidates[0]
        result.score, result.margin = score, score-candidates[1][0]
        if score < match['min_score'] or result.margin < match['min_margin']:
            result.reason = '槽形匹配不足或四个方向无法可靠区分'
            return result
        frame_coverage = float((self._warp(valid_image, pose, True)>0).mean())
        if frame_coverage < match['min_frame_coverage']:
            result.reason = '料盘未完整入画'
            return result
        result.orientation_ok, result.pose = True, pose
        pose.used = ['tray_slot_template']
        metal = self._warp(metal_mask, pose, True)>0
        brass = self._warp(brass_mask, pose, True)>0 if brass_mask is not None else None
        occupied_evidence = metal | brass if evidence == 'metal_or_brass' else metal
        hand = self._warp(glove, pose, True)>0
        thresholds = self.settings['occupancy']
        for sid, mask in self.slot_masks.items():
            coverage = float(occupied_evidence[mask].mean())
            occlusion = float(hand[mask].mean())
            if occlusion > thresholds['max_glove_fraction']:
                state = 'unknown'
            elif coverage >= thresholds.get('present_fraction_by_slot', {}).get(sid, thresholds['present_fraction']):
                state = 'occupied'
            elif coverage <= thresholds['empty_fraction']:
                state = 'empty'
            else:
                state = 'unknown'
            result.slots[sid] = {'state': state, 'metal_fraction': float(metal[mask].mean()),
                                 'brass_fraction': float(brass[mask].mean()) if brass is not None else None,
                                 'occupancy_fraction': coverage, 'evidence': evidence,
                                 'glove_fraction': occlusion}
        states = [row['state'] for row in result.slots.values()]
        if 'unknown' in states:
            result.reason = '存在遮挡或占用证据处于不确定带'
        elif all(s == 'empty' for s in states):
            result.state = 'empty'
        elif all(s == 'occupied' for s in states):
            result.state = 'full'
        else:
            result.state = 'partial'
        return result
