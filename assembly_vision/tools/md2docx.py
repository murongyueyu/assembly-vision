# -*- coding: utf-8 -*-
"""把本次的 Markdown 素材转成 Word（报告用）。

只支持本文档用到的子集：标题、段落、引用、无序/有序列表、表格、代码块、行内 **粗体** 与 `代码`。
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

CJK_FONT = '微软雅黑'
MONO_FONT = 'Consolas'
BODY_PT = 10.5


def set_cjk(run, name=CJK_FONT):
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)


def add_rich(paragraph, text, size=BODY_PT, mono=False):
    """处理 **粗体** 与 `代码` 两种行内标记。"""
    for part in re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
            set_cjk(run)
        elif part.startswith('`') and part.endswith('`'):
            run = paragraph.add_run(part[1:-1])
            run.font.name = MONO_FONT
            run.font.size = Pt(size - 1)
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        else:
            run = paragraph.add_run(part)
            set_cjk(run)
        run.font.size = Pt(size if not mono else size - 1)


def build(md_path, out_path):
    lines = Path(md_path).read_text(encoding='utf-8').splitlines()
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = CJK_FONT
    style.font.size = Pt(BODY_PT)
    style.element.rPr.rFonts.set(qn('w:eastAsia'), CJK_FONT)

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        if line.startswith('```'):
            i += 1
            block = []
            while i < n and not lines[i].startswith('```'):
                block.append(lines[i])
                i += 1
            i += 1
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Pt(12)
            for k, code in enumerate(block):
                run = para.add_run(code if k == 0 else '\n' + code)
                run.font.name = MONO_FONT
                run.font.size = Pt(9)
            continue

        m = re.match(r'^(#{1,4})\s+(.*)$', line)
        if m:
            level = len(m.group(1))
            text = m.group(2)
            if level == 1:
                h = doc.add_heading('', level=0)
            else:
                h = doc.add_heading('', level=level - 1)
            add_rich(h, text, size={1: 18, 2: 14, 3: 12, 4: 11}.get(level, 11))
            for run in h.runs:
                run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            i += 1
            continue

        if line.startswith('|') and i + 1 < n and re.match(r'^\|[\s:|-]+\|$', lines[i + 1].strip()):
            header = [c.strip() for c in line.strip().strip('|').split('|')]
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = 'Table Grid'
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for j, text in enumerate(header):
                cell = table.rows[0].cells[j]
                cell.text = ''
                para = cell.paragraphs[0]
                add_rich(para, text, size=9.5)
                for run in para.runs:
                    run.bold = True
            for row in rows:
                cells = table.add_row().cells
                for j, text in enumerate(row[:len(header)]):
                    cells[j].text = ''
                    add_rich(cells[j].paragraphs[0], text, size=9.5)
            doc.add_paragraph()
            continue

        if line.startswith('> '):
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Pt(18)
            add_rich(para, line[2:], size=9.5)
            for run in para.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
            i += 1
            continue

        m = re.match(r'^!\[(.*?)\]\((.+?)\)$', line.strip())
        if m:
            from docx.shared import Cm
            caption, rel = m.group(1), m.group(2)
            img_path = (Path(md_path).parent / rel).resolve()
            if img_path.exists():
                pic = doc.add_paragraph()
                pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pic.add_run().add_picture(str(img_path), width=Cm(15.5))
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cap.add_run(caption)
                run.italic = True
                run.font.size = Pt(9)
                set_cjk(run)
            else:
                para = doc.add_paragraph()
                add_rich(para, '（缺图：%s）' % rel, size=9)
            i += 1
            continue

        m = re.match(r'^[-*]\s+(.*)$', line)
        if m:
            para = doc.add_paragraph(style='List Bullet')
            add_rich(para, m.group(1))
            i += 1
            continue

        m = re.match(r'^(\d+)\.\s+(.*)$', line)
        if m:
            para = doc.add_paragraph(style='List Number')
            add_rich(para, m.group(2))
            i += 1
            continue

        para = doc.add_paragraph()
        para.paragraph_format.space_after = Pt(4)
        add_rich(para, line)
        i += 1

    doc.save(out_path)
    return out_path


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else 'docs/2026-10-29_项目实现说明_报告素材.md'
    dst = sys.argv[2] if len(sys.argv) > 2 else '2026-10-29_ABCDE_项目实现说明.docx'
    print('written:', build(src, dst))
