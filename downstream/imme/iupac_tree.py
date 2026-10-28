# -*- coding: utf-8 -*-
"""IUPAC condensed 糖链解析与树同构比较

本地独立验证，不依赖API：
- parse_iupac: 把 IUPAC condensed 字符串解析为残基树
- signature:  树的规范化签名（分支顺序无关，同构<=>
签名相等）
- compare_iupac: 逐项核对残基、连接、修饰，返回差异描述
- validate_iupac: 输入合法性检查（括号配对/基本语法），不合法则不送API
"""

import re
from collections import Counter

# API回转时的等价命名漂移（按需补充）
RESIDUE_ALIASES = {
    'FucA': 'L-GalA',
}

# 残基名token的合法字符（含占位符?、构型前缀-、数字开头如6dTal）
_TOKEN_RE = re.compile(r'[A-Za-z0-9?\-]+')


class ParseError(Exception):
    pass


class Node:
    __slots__ = ('name', 'link', 'children')

    def __init__(self, name, link=None):
        self.name = name    # 残基名（含内嵌修饰，如 GlcNAc、Galf?Ac、Man?P）
        self.link = link    # 连接信息（如 'a1-3'），还原端为 None
        self.children = []  # 子树列表（无序，比较时按规范化签名配对）


def _match_brackets(s):
    """返回 ({开: 闭}, {闭: 开})；括号不配对抛 ParseError"""
    pairs = {}
    stack = []
    for i, ch in enumerate(s):
        if ch == '[':
            stack.append(i)
        elif ch == ']':
            if not stack:
                raise ParseError(f"多余的']'，位置{i}")
            pairs[stack.pop()] = i
    if stack:
        raise ParseError(f"'['未闭合，位置{stack[0]}")
    return pairs, {v: k for k, v in pairs.items()}


def _parse_chain(s, i, j, pairs, rev_pairs):
    """解析 s[i:j] 为一条链（根=最右残基），返回根Node"""
    if i >= j:
        raise ParseError("空链")

    # 根残基可选的连接信息：区间末尾是 '(...)'（分支链根必带，还原端不带）
    link = None
    core_end = j
    if s[j - 1] == ')':
        depth = 0
        k = j - 1
        while k >= i:
            if s[k] == ')':
                depth += 1
            elif s[k] == '(':
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        if k < i:
            raise ParseError(f"连接括号不配对: {s[i:j]}")
        link = s[k + 1:j - 1].strip()
        core_end = k

    # 根残基名：core_end 左侧的最右token
    m = None
    for m in _TOKEN_RE.finditer(s, i, core_end):
        pass
    if m is None or m.end() != core_end:
        raise ParseError(f"无法解析残基: {s[i:j]}")
    root = Node(m.group(0), link)

    # 处理 elements 区间 [i, m.start())，从右向左扫描：
    # 分支[...] 挂在当前链头残基上（链头不变）；链残基 RESIDUE(LINK) 挂链头并成为新链头
    p = m.start()
    current = root
    while p > i:
        if s[p - 1] == ']':
            k = rev_pairs.get(p - 1)
            if k is None or k < i:
                raise ParseError(f"分支括号错位: 位置{p - 1}")
            child = _parse_chain(s, k + 1, p - 1, pairs, rev_pairs)
            current.children.append(child)
            p = k
        else:
            m2 = re.search(r'([A-Za-z0-9?\-]+)\(([^()]*)\)$', s[i:p])
            if not m2:
                raise ParseError(f"无法解析链元素: {s[i:p][-40:]}")
            child = Node(m2.group(1), m2.group(2).strip())
            current.children.append(child)
            current = child
            p = i + m2.start()
    return root


def parse_iupac(s):
    """解析IUPAC condensed字符串为残基树，返回根节点；失败抛ParseError"""
    s = s.strip()
    if not s:
        raise ParseError("空字符串")
    pairs, rev_pairs = _match_brackets(s)
    return _parse_chain(s, 0, len(s), pairs, rev_pairs)


def normalize_residue(name):
    """残基命名规范化：处理API回转时的等价命名漂移"""
    for alias in sorted(RESIDUE_ALIASES, key=len, reverse=True):
        if name.startswith(alias):
            return RESIDUE_ALIASES[alias] + name[len(alias):]
    return name


def signature(node):
    """树的规范化签名：分支排序无关，两树同构 <=> 签名相等"""
    child_sigs = sorted(signature(c) for c in node.children)
    return f"{normalize_residue(node.name)}<{node.link or ''}>[{','.join(child_sigs)}]"


def _subtree_brief(node):
    """子树摘要（用于差异报告）"""
    return f"{node.name}({node.link})" if node.link else node.name


def _diff(node_m, node_v, path, out):
    """递归比较两树，差异写入out。前提：两树签名不等"""
    nm, nv = normalize_residue(node_m.name), normalize_residue(node_v.name)
    if nm != nv:
        out.append(f"残基@{path}: {node_m.name}!={node_v.name}")
    lm, lv = node_m.link or '', node_v.link or ''
    if lm != lv:
        out.append(f"连接@{path}: {lm}!={lv}")

    # 子树配对：先按签名精确配对（无差异），剩余按根残基名配对后递归
    unmatched_v = list(node_v.children)
    for cm in node_m.children:
        sig = signature(cm)
        hit = next((cv for cv in unmatched_v if signature(cv) == sig), None)
        if hit is not None:
            unmatched_v.remove(hit)
            continue
        # 签名不等：找根残基同名的做近似配对，递归报具体差异
        near = next((cv for cv in unmatched_v
                     if normalize_residue(cv.name) == normalize_residue(cm.name)), None)
        if near is not None:
            unmatched_v.remove(near)
            _diff(cm, near, f"{path}/{_subtree_brief(cm)}", out)
        else:
            out.append(f"缺失分支@{path}: {_subtree_brief(cm)}(含{len(signature(cm))}字符子树)")
    for cv in unmatched_v:
        out.append(f"多出分支@{path}: {_subtree_brief(cv)}")


def compare_iupac(modified, verified):
    """比较两个IUPAC condensed结构的树同构性

    返回 (status, detail):
      'OK'          同构
      'DIFF'        结构差异，detail为逐项差异描述
      'PARSE_FAIL'  解析失败，detail为原因
    """
    try:
        tree_m = parse_iupac(modified)
    except ParseError as e:
        return 'PARSE_FAIL', f'modified解析失败: {e}'
    try:
        tree_v = parse_iupac(verified)
    except ParseError as e:
        return 'PARSE_FAIL', f'verified解析失败: {e}'
    if signature(tree_m) == signature(tree_v):
        return 'OK', ''
    out = []
    _diff(tree_m, tree_v, normalize_residue(tree_m.name), out)
    return 'DIFF', '; '.join(out[:5])


def validate_iupac(s):
    """输入合法性检查（能否建树）。返回None=合法，否则为错误原因"""
    try:
        parse_iupac(s)
        return None
    except ParseError as e:
        return str(e)
