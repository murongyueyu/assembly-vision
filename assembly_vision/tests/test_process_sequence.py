# -*- coding: utf-8 -*-
"""E顺序推定：新帧确认、遮挡、越序、多件、放回、中断、复位。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assemvision.config import load_config
from assemvision.core.checks import EvidenceStamp, PrestartResult, PrestartSlot
from assemvision.core.fsm import PickSequence

ROOT = Path(__file__).resolve().parents[1]


class Scenario:
    def __init__(self):
        self.net = PickSequence(load_config(str(ROOT/'config/workpiece.yaml')),
                               load_config(str(ROOT/'config/process_sequence.yaml')).data)
        self.states = {sid: 'match' for sid in self.net.expected}
        self.t, self.index = 100., 0

    def feed(self, n=1, allow=False, valid=True, gap=.16):
        for _ in range(n):
            self.t += gap
            self.index += 1
            self.result = PrestartResult(EvidenceStamp(self.index, self.t, (480,640)),
                {sid: PrestartSlot(pid, observed=pid if self.states[sid]=='match' else '',
                                  state=self.states[sid]) for sid,pid in self.net.expected.items()},
                allow_start=allow, valid=valid)
            verdict = self.net.update(self.result, self.t)
        return verdict

    def start(self):
        self.feed(allow=True)
        assert self.net.started and not self.net.done


def main():
    s = Scenario()
    s.feed(5)
    assert not s.net.started
    s.start()
    for i, sid in enumerate(s.net.order):
        s.states[sid] = 'unknown'
        s.feed(5)
        assert len(s.net.done) == i
        s.states[sid] = 'missing'
        s.feed(2)
        assert len(s.net.done) == i
        for _ in range(5):
            s.net.update(s.result, s.t)
        assert len(s.net.done) == i
        v = s.feed()
        assert len(s.net.done) == i+1
    assert v.step == 4 and '推定完成' in v.title
    s.feed(4)
    assert len(s.net.done)==4
    s.states[s.net.order[1]]='match'
    s.feed(3)
    assert len(s.net.done)==1
    s.feed(3)
    assert len(s.net.done)==1
    assert '越序' in s.net.verdict.title
    s.net.reset(); assert not s.net.started and not s.net.events
    s.feed(3); assert not s.net.started

    s=Scenario();s.start()
    s.states[s.net.order[2]]='missing'
    assert s.feed(3).level=='error' and not s.net.done
    s.states[s.net.order[2]]='match';s.feed(3)
    s.states[s.net.order[0]]='missing';s.states[s.net.order[1]]='missing'
    assert s.feed(3).level=='error' and not s.net.done
    for sid in s.states:s.states[sid]='match'
    s.feed(3)
    s.states[s.net.order[0]]='missing';s.feed(3)
    assert len(s.net.done)==1

    for failure in ('invalid', 'gap', 'stale', 'reverse', 'duplicate_changed'):
        s=Scenario();s.start()
        if failure=='invalid':s.feed(valid=False)
        elif failure=='gap':s.feed(gap=1.)
        elif failure=='stale':s.net.update(s.result,s.t+2)
        elif failure=='reverse':
            s.result.stamp=EvidenceStamp(0,s.t-.1,(480,640));s.net.update(s.result,s.t)
        else:
            s.result.slots[s.net.order[0]].state='missing';s.net.update(s.result,s.t)
        s.states[s.net.order[0]]='missing';s.feed(4)
        assert not s.net.done, failure
        s.states[s.net.order[0]]='match';s.feed(3)
        s.states[s.net.order[0]]='missing';s.feed(3)
        assert len(s.net.done)==1, failure
    s=Scenario();s.start();s.feed(gap=16)
    assert s.net.verdict.level=='warn'
    # 错件与未知不能靠其他槽的取件把流程带过去；必须恢复连续当前证据。
    s=Scenario();s.start()
    s.states[s.net.order[0]]='missing'
    s.states[s.net.order[3]]='wrong'
    assert s.feed(4).level=='error' and not s.net.done
    s.states[s.net.order[3]]='unknown'
    s.feed(4);assert not s.net.done
    s.states[s.net.order[3]]='match'
    s.feed(2);assert not s.net.done
    s.feed();assert len(s.net.done)==1
    print('E流程：四步、遮挡、重复帧、越序/多取、放回回退及五种中断恢复通过')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
