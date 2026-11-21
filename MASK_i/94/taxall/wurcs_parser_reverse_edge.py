# -*- coding: utf-8 -*-
"""
WURCS to GNN Graph Parser
完全独立的WURCS格式解析器，将WURCS转换为GNN输入格式
"""
import re
import sys
import csv
try:
    from scipy.stats import binomtest
except ImportError:
    from scipy.stats import binom_test as binomtest
from concurrent.futures import ThreadPoolExecutor, as_completed

class WURCSParser:
    """WURCS格式解析器"""

    def __init__(self):
        pass

    def parse(self, WURCS):
        """
        解析WURCS字符串，返回GNN图结构
        """
        result = {
            "metadata": {},
            "big_nodes": [],
            "small_nodes": [],
            "big_edges": [],
            "small_edges": []
        }

        if not WURCS.startswith("WURCS="):
            raise ValueError("Invalid WURCS format: must start with 'WURCS='")

        remaining = WURCS[6:]

        first_slash = remaining.find('/')
        version = remaining[:first_slash]
        after_version = remaining[first_slash + 1:]

        second_slash = after_version.find('/')
        metadata_str = after_version[:second_slash]
        after_metadata = after_version[second_slash + 1:]

        metadata_parts = metadata_str.split(",")
        if len(metadata_parts) != 3:
            raise ValueError("Invalid WURCS format: metadata should have 3 parts")

        template_count = int(metadata_parts[0])
        sugar_count = int(metadata_parts[1])
        linkage_count = int(metadata_parts[2])

        result["metadata"] = {
            "template_count": template_count,
            "sugar_count": sugar_count,
            "linkage_count": linkage_count
        }

        last_bracket_pos = -1
        bracket_depth = 0
        for i, char in enumerate(after_metadata):
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    last_bracket_pos = i + 1

        if last_bracket_pos == -1:
            raise ValueError("Invalid WURCS format: cannot find template end")

        templates_str = after_metadata[:last_bracket_pos]
        after_templates = after_metadata[last_bracket_pos:]

        if not after_templates.startswith('/'):
            raise ValueError("Invalid WURCS format: no slash after templates. Got: %s" % after_templates)
        after_templates = after_templates[1:]

        third_slash = after_templates.find('/')
        if third_slash == -1:
            if linkage_count == 0:
                index_str = after_templates
                linkage_str = ""
            else:
                raise ValueError("Invalid WURCS format: no slash after index")
        else:
            index_str = after_templates[:third_slash]
            linkage_str = after_templates[third_slash + 1:]

        # 计算连接区中}*出现的次数（}*取代基不算真正的糖苷键连接）
        brace_star_count = linkage_str.count("}*")
        true_linkage_count = linkage_count - brace_star_count
        result["metadata"]["true_linkage_count"] = true_linkage_count

        templates = re.findall(r'\[([^\]]+)\]', templates_str)

        indices = [int(x) for x in index_str.split("-")]

        # 智能分割linkage：保护 }*取代基 中的 / 不被 _ 切断
        # 例如 a?|b?}*OSO/3=O/3=O 应该是一个整体，不能被 _ 切成 a?|b?}*OSO/ 和 3=O
        # 关键：取代基中不含 _，_ 始终是linkage的分隔符
        # 需要保护的是 / 后面的内容不被 _ 切断
        # 正确分割：a?|b?}*OSO/3=O/3=O 是一个整体，下一个 _ 才是新的linkage
        linkages = []
        current = ""
        in_substituent = False  # 是否在 }*取代基 保护模式中（保护 / 后的内容）
        pending_brace = False   # 遇到 } 后等待判断是否跟 *
        for ch in linkage_str:
            if pending_brace:
                if ch == '*':
                    # }* 确认，进入取代基保护模式
                    current += ch
                    pending_brace = False
                    in_substituent = True
                else:
                    # } 后面不是 *，正常处理
                    current += ch
                    pending_brace = False
                    if ch == '_':
                        linkages.append(current[:-1])
                        current = ""
                    # }- 是正常的双端或关系，不在保护模式
            elif in_substituent:
                # 在取代基保护模式中，/ 后面的数字和=不被 _ 切断
                # 但 _ 始终是linkage分隔符，取代基中不含 _
                if ch == '_':
                    # _ 结束取代基保护模式，也是linkage分隔符
                    linkages.append(current)
                    current = ""
                    in_substituent = False
                elif ch == '-':
                    # - 表示取代基结束，后面是连接目标，仍在同一个linkage中
                    current += ch
                    in_substituent = False
                elif ch == '{':
                    # { 表示取代基结束，后面是双端或关系的目标端
                    current += ch
                    in_substituent = False
                else:
                    current += ch
            else:
                if ch == '}':
                    current += ch
                    pending_brace = True
                elif ch == '_':
                    linkages.append(current)
                    current = ""
                else:
                    current += ch
        if current:
            linkages.append(current)

        def _parse_template(template_str):
            if template_str.startswith("<Q>"):
                # <Q>未知模板，可能带异头信息如 <Q>-?a
                anom_pos = "unknown"
                anom_conf = "unknown"
                rest = template_str[3:]  # 去掉<Q>
                if rest.startswith('-'):
                    match = re.match(r'-(\d+|\?)([abxABx])', rest)
                    if match:
                        pos = match.group(1)
                        anom_pos = int(pos) if pos.isdigit() else pos
                        anom_conf = match.group(2).lower()
                # 有异头碳信息则认为成环
                has_ring = True if anom_pos != "unknown" else "unknown"
                return {
                    "stereochemistry": "unknown",
                    "carbon_count": "unknown",
                    "ring_bonds": [{"ring_bond_type": "unknown", "probability": 1}],
                    "has_ring": has_ring,
                    "anomeric_position": anom_pos,
                    "anomeric_configuration": anom_conf,
                    "substituents": {},
                    "is_unknown": True
                }

            carbon_count = 6
            ring_bonds = []
            anomeric_position = "N/A"
            anomeric_configuration = "N/A"
            stereochemistry = ""
            substituents = {}
            double_bonds = []  # 新增：存储双键位置，如 ["2=3"]

            base_end = len(template_str)
            for char in ['-', '_']:
                pos = template_str.find(char)
                if pos != -1 and pos < base_end:
                    base_end = pos

            stereochemistry = template_str[:base_end]
            carbon_count = len(stereochemistry)

            # 根据立体化学信息中的字符确定取代基
            # 双键字符：E、F、e、f、Z、z、n、N 都表示双键位置
            # n/N 是双键顺反不确定（neither E nor Z）
            double_bond_chars = {'E', 'F', 'e', 'f', 'Z', 'z', 'n', 'N'}
            
            for i, char in enumerate(stereochemistry):
                carbon_pos = i + 1  # 碳位从1开始
                if char in ['d', 'm']:
                    # d, m 对应的取代基为 H（脱氧）
                    substituents[carbon_pos] = 'H'
                elif char in double_bond_chars:
                    # 双键字符对应的取代基为 none（表示没有取代基，实际是氢）
                    substituents[carbon_pos] = 'none'

            # 检测双键位置：连续的双键字符表示双键，如 ZZ 表示位置2=3之间有双键
            # 双键字符成对出现：Ee/Ez/Zz/EZ/nn/Nn 等
            # 每对连续的双键字符表示相邻两个碳之间有一个双键
            i = 0
            while i < len(stereochemistry):
                char = stereochemistry[i]
                if char in double_bond_chars:
                    # 找到双键起始位置
                    j = i + 1
                    while j < len(stereochemistry) and stereochemistry[j] in double_bond_chars:
                        j += 1
                    # 双键跨越的位置：i+1 = j (i+1到j之间是双键)
                    if j > i + 1:
                        double_bonds.append(f"{i+1}={j}")
                    i = j
                else:
                    i += 1

            modifiers = template_str[base_end:].split("_")

            for mod in modifiers:
                if not mod:
                    continue

                if mod.startswith('-'):
                    match = re.match(r'-(\d+|\?)([abxABx])', mod)
                    if match:
                        pos = match.group(1)
                        anomeric_position = int(pos) if pos.isdigit() else pos
                        anomeric_configuration = match.group(2).lower()

                elif re.match(r'^[\d?]+-[\d?]+', mod):
                    # 解析环信息，展开为ring_bonds列表
                    # 每条ring_bond: {start, end, probability, linker}
                    # 支持 start-end、start-end*linker、start-endlinker*subinfo 格式
                    # linker 表示环键通过什么相连，默认为 O
                    # 如 1-5 表示1-5通过O成环
                    # 如 4-6OPO*/3O/3=O 表示4-6通过OPO*/3O/3=O成环
                    linker = "O"
                    ring_part = mod
                    # 先提取纯数字的环键部分 start-end
                    ring_match = re.match(r'^([\d?]+)-([\d?]+)(.*)', mod)
                    if ring_match:
                        start_str = ring_match.group(1)
                        end_str = ring_match.group(2)
                        remainder = ring_match.group(3)
                        # remainder 可能是 *linker 或 linker*subinfo 或空
                        if remainder.startswith('*'):
                            # 如 1-5*OPO 表示1-5通过OPO成环
                            linker = remainder[1:]
                        elif remainder:
                            # 如 4-6OPO*/3O/3=O 表示4-6通过OPO*/3O/3=O成环
                            linker = remainder
                        # else: 无remainder，linker默认为O
                    else:
                        if '*' in mod:
                            ring_part, linker = mod.split('*', 1)
                        parts = ring_part.split('-')
                        start_str, end_str = parts[0], parts[1]
                    start = int(start_str) if start_str.isdigit() else start_str
                    end = int(end_str) if end_str.isdigit() else end_str

                    # 如果 linker 不是 O，将 linker 作为两个位置的取代基
                    if linker != "O":
                        if isinstance(start, int):
                            substituents[start] = linker
                        if isinstance(end, int):
                            substituents[end] = linker

                    # 构建 ring_bond_type: "start-end" (简化，不包含linker)
                    def _make_ring_bond_type(s, e):
                        return "%s-%s" % (s, e)

                    ring_bond_type_str = "%s-%s" % (start, end)
                    ring_bonds.append({
                        "ring_bond_type": ring_bond_type_str,
                        "probability": 1
                    })
                    # 将 linker 作为取代基
                    if linker != "O":
                        substituents[start] = linker
                        substituents[end] = linker

                elif '*' in mod:
                    match = re.match(r'(\d+)\*(.+)', mod)
                    if match:
                        position = int(match.group(1))
                        substituent = match.group(2)
                        substituents[position] = substituent
                    else:
                        # 处理 ?*OPO/3O/3=O 这种不确定位置的取代基
                        # 支持多个?*取代基，用列表存储
                        match = re.match(r'([?])\*(.+)', mod)
                        if match:
                            substituent = match.group(2)
                            if '?' not in substituents:
                                substituents['?'] = []
                            substituents['?'].append(substituent)

            # 如果没有双键，设为["none"]
            if not double_bonds:
                double_bonds = ["none"]

            # 判断是否成环：有成环信息 或 有异头碳 都算成环
            if isinstance(ring_bonds, list) and len(ring_bonds) > 0:
                has_ring = True
            elif anomeric_position != "N/A":
                # 有异头碳但无成环信息，如 <Q>-?a
                has_ring = True
            else:
                # 无成环信息，也无异头碳，表示不成环
                has_ring = False
                ring_bonds = [{"ring_bond_type": "N/A", "probability": 1}]
            return {
                "stereochemistry": stereochemistry,
                "carbon_count": carbon_count,
                "ring_bonds": ring_bonds,
                "has_ring": has_ring,
                "anomeric_position": anomeric_position,
                "anomeric_configuration": anomeric_configuration,
                "substituents": substituents,
                "double_bonds": double_bonds,  # 新增：双键位置
                "is_unknown": False
            }

        parsed_templates = [_parse_template(t) for t in templates]

        big_nodes = []
        for sugar_idx in range(sugar_count):
            template_idx = indices[sugar_idx] - 1
            if template_idx >= len(parsed_templates):
                raise ValueError("Invalid WURCS format: template index out of range")
            template = parsed_templates[template_idx]
            # 构建 big_node 特征
            rb = template["ring_bonds"]

            # carbon_count_type: 用于 embedding
            carbon_count_type = str(template["carbon_count"])

            # anomeric_position_type: 用于 embedding
            anomeric_position_type = str(template["anomeric_position"])

            # anomeric_config_type: 用于 embedding
            anomeric_config_type = str(template["anomeric_configuration"])

            # double_bonds: 双键位置列表
            double_bonds = template.get("double_bonds", [])

            big_nodes.append({
                "id": sugar_idx,
                "carbon_count_type": carbon_count_type,
                "ring_bonds": rb,
                "anomeric_position_type": anomeric_position_type,
                "anomeric_config_type": anomeric_config_type,
                "double_bonds": double_bonds  # 新增：双键位置特征
            })

        result["big_nodes"] = big_nodes

        # 预解析linkages，收集被大边占用的位置
        occupied_positions = {}  # {sugar_idx: set of occupied positions}

        char_to_sugar_idx = {}
        for i_sugar in range(sugar_count):
            if i_sugar < 26:
                char = chr(ord('a') + i_sugar)
            elif i_sugar < 52:
                char = chr(ord('A') + (i_sugar - 26))
            else:
                j = i_sugar - 52
                first = chr(ord('a') + (j // 52))
                second_idx = j % 52
                if second_idx < 26:
                    second = chr(ord('a') + second_idx)
                else:
                    second = chr(ord('A') + (second_idx - 26))
                char = first + second
            char_to_sugar_idx[char] = i_sugar

        def _collect_occupied_positions(linkage_str, char_to_sugar_idx, occupied_positions):
            """从linkage字符串中收集被占用的位置"""
            if not linkage_str:
                return

            # 用下划线分割
            for part in linkage_str.split('_'):
                if not part:
                    continue
                
                # 用-分割source和target
                if '-' not in part:
                    continue
                dash_pos = part.index('-')
                source_part = part[:dash_pos]
                target_part = part[dash_pos + 1:]

                # 解析source（即使整个part包含大括号，source部分依然要收集）
                # source不应包含大括号，直接匹配
                source_match = re.match(r'([a-zA-Z]{1,2})([\d?]+)', source_part)
                if source_match:
                    sugar_char = source_match.group(1)
                    pos_str = source_match.group(2)
                    # 位置中有?号的不要收集
                    if pos_str.isdigit() and '?' not in pos_str:
                        sugar_idx = char_to_sugar_idx.get(sugar_char)
                        if sugar_idx is not None:
                            if sugar_idx not in occupied_positions:
                                occupied_positions[sugar_idx] = set()
                            occupied_positions[sugar_idx].add(int(pos_str))

                # 解析target（跳过target包含大括号或?号的情况）
                if '{' in target_part or '}' in target_part or '?' in target_part:
                    continue
                target_match = re.match(r'([a-zA-Z]{1,2})([\d?]+)', target_part)
                if target_match:
                    sugar_char = target_match.group(1)
                    pos_str = target_match.group(2)
                    # 位置中有?号的不要收集
                    if pos_str.isdigit() and '?' not in pos_str:
                        sugar_idx = char_to_sugar_idx.get(sugar_char)
                        if sugar_idx is not None:
                            if sugar_idx not in occupied_positions:
                                occupied_positions[sugar_idx] = set()
                            occupied_positions[sugar_idx].add(int(pos_str))

        # 收集所有被大边占用的位置
        for linkage in linkages:
            if not linkage:
                continue
            _collect_occupied_positions(linkage, char_to_sugar_idx, occupied_positions)
        # 收集环占用的位置
        # 环键位置同样算占用：不确定位置的取代基不能落在环键位置上去抢环上的O
        for sugar_idx in range(sugar_count):
            template_idx = indices[sugar_idx] - 1
            template = parsed_templates[template_idx]
            for rb in template.get("ring_bonds", []):
                rb_type = rb.get("ring_bond_type", "")
                if "-" in rb_type:
                    parts = rb_type.split("-")
                    start_pos = parts[0]
                    end_pos = parts[1]
                    # 只收集确定的数字位置，?号不收集
                    if start_pos.isdigit():
                        if sugar_idx not in occupied_positions:
                            occupied_positions[sugar_idx] = set()
                        occupied_positions[sugar_idx].add(int(start_pos))
                    if end_pos.isdigit():
                        if sugar_idx not in occupied_positions:
                            occupied_positions[sugar_idx] = set()
                        occupied_positions[sugar_idx].add(int(end_pos))
        small_nodes = []
        small_edges = []
        small_node_id = 0
        # 记录所有由"不确定位置取代基"产生的边的下标，供最后统一调整O边概率
        doubt_edge_indices = []

        # 记录每个sugar的碳位置对应的小节点ID，用于doubt边
        sugar_carbon_small_node_ids = {}

        for sugar_idx in range(sugar_count):
            template_idx = indices[sugar_idx] - 1
            template = parsed_templates[template_idx]

            if template.get("is_unknown", False):
                continue

            sugar_carbon_small_node_ids[sugar_idx] = {}

            # 判断是否有不确定位置的取代基
            doubt_substituents = template["substituents"].get('?', [])
            num_doubt = len(doubt_substituents)
            carbon_total = template["carbon_count"]

            # 计算O小节点数量 = carbon_total - 确定非O取代基数（包括none）
            # 不确定位置的取代基不扣除O节点：O先全部建立，再由doubt取代基按概率"掠夺"
            # 'none' 现在是一个有效的取代基，需要创建小节点
            num_firm_non_O = len([p for p in range(1, carbon_total + 1) if p in template["substituents"] and template["substituents"][p] != 'O'])
            num_O_nodes = carbon_total - num_firm_non_O

            # 创建确定位置的非O取代基小节点（包括none）
            firm_small_node_ids = {}
            for carbon_pos in range(1, carbon_total + 1):
                if carbon_pos in template["substituents"]:
                    substituent = template["substituents"][carbon_pos]
                    # 'none' 也是有效的取代基，创建小节点
                    small_nodes.append({
                        "id": small_node_id,
                        "substituent": substituent
                    })
                    firm_small_node_ids[carbon_pos] = small_node_id
                    small_node_id += 1

            # 创建O小节点
            o_small_node_ids = []
            for i in range(num_O_nodes):
                small_nodes.append({
                    "id": small_node_id,
                    "substituent": "O"
                })
                o_small_node_ids.append(small_node_id)
                small_node_id += 1

            # 创建doubt取代基小节点
            doubt_small_node_ids = []
            for substituent in doubt_substituents:
                small_nodes.append({
                    "id": small_node_id,
                    "substituent": substituent
                })
                doubt_small_node_ids.append(small_node_id)
                small_node_id += 1

            # 构建small_edges
            o_idx = 0  # O小节点索引
            for carbon_pos in range(1, carbon_total + 1):
                # 检查该位置是否有确定的非O取代基（包括none）
                if carbon_pos in firm_small_node_ids:
                    small_edges.append({
                        "big_node_id": sugar_idx,
                        "small_node_id": firm_small_node_ids[carbon_pos],
                        "stereo_type": template["stereochemistry"][carbon_pos - 1],
                        "carbon_position_type": str(carbon_pos),
                        "probability": 1
                    })
                else:
                    # 使用O小节点
                    if o_idx < len(o_small_node_ids):
                        small_edges.append({
                            "big_node_id": sugar_idx,
                            "small_node_id": o_small_node_ids[o_idx],
                            "stereo_type": template["stereochemistry"][carbon_pos - 1],
                            "carbon_position_type": str(carbon_pos),
                            "probability": 1
                        })
                        o_idx += 1

            # 添加doubt边（不确定位置的取代基）
            if doubt_small_node_ids:
                # 获取所有没有确定取代基的碳位置
                available_positions = []
                sugar_occupied = occupied_positions.get(sugar_idx, set())
                for carbon_pos in range(1, carbon_total + 1):
                    if carbon_pos not in firm_small_node_ids and carbon_pos not in sugar_occupied:
                        available_positions.append(carbon_pos)

                # 如果有可用位置，创建doubt边
                if available_positions:
                    prob = 1.0 / len(available_positions)
                    for doubt_node_id in doubt_small_node_ids:
                        for pos in available_positions:
                            doubt_edge_indices.append(len(small_edges))
                            small_edges.append({
                                "big_node_id": sugar_idx,
                                "small_node_id": doubt_node_id,
                                "stereo_type": template["stereochemistry"][pos - 1],
                                "carbon_position_type": str(pos),
                                "probability": prob
                            })

            # 记录小节点ID映射
            sugar_carbon_small_node_ids[sugar_idx] = firm_small_node_ids

        result["small_nodes"] = small_nodes
        result["small_edges"] = small_edges

        # 解析连接信息, 构建big_edges
        # 移植自古老版本的概率体系, 同时保留反向边
        big_edges = []

        # 构建糖标识符到索引的映射
        # 0-25: a-z, 26-51: A-Z, 52+: 双字母(aa,ab,...,aA,aB,...,ba,bb,...)
        char_to_sugar_idx = {}
        for i_sugar in range(sugar_count):
            if i_sugar < 26:
                # 1-26: a-z (单小写字母)
                char = chr(ord('a') + i_sugar)
            elif i_sugar < 52:
                # 27-52: A-Z (单大写字母)
                char = chr(ord('A') + (i_sugar - 26))
            else:
                # 53+: 双字母编码
                j = i_sugar - 52
                first = chr(ord('a') + (j // 52))
                second_idx = j % 52
                if second_idx < 26:
                    second = chr(ord('a') + second_idx)
                else:
                    second = chr(ord('A') + (second_idx - 26))
                char = first + second
            char_to_sugar_idx[char] = i_sugar

        def _parse_single_linkage(link_str):
            """解析单个linkage, 如 a3-b1, 返回edge字典; 解析失败则抛出ValueError"""
            link_match = re.match(r'([a-zA-Z]{1,2})([\d?]+)(?:\*([^_|]+))?-([a-zA-Z]{1,2})([\d?]+)(?:\*([^_|]+))?', link_str)
            if not link_match:
                raise ValueError("Invalid linkage format: cannot parse '%s'" % link_str)

            source_char = link_match.group(1)
            source_pos_str = link_match.group(2)
            source_pos = int(source_pos_str) if source_pos_str.isdigit() else source_pos_str
            source_substituent = link_match.group(3)
            target_char = link_match.group(4)
            target_pos_str = link_match.group(5)
            target_pos = int(target_pos_str) if target_pos_str.isdigit() else target_pos_str
            target_substituent = link_match.group(6)

            source_sugar_idx = char_to_sugar_idx.get(source_char)
            if source_sugar_idx is None:
                raise ValueError("Invalid linkage '%s': source sugar '%s' not found" % (link_str, source_char))

            target_sugar_idx = char_to_sugar_idx.get(target_char)
            if target_sugar_idx is None:
                raise ValueError("Invalid linkage '%s': target sugar '%s' not found" % (link_str, target_char))

            biglink = source_substituent if source_substituent else (target_substituent if target_substituent else "O")

            base = "%s-%s" % (source_pos, target_pos)
            big_edge_type = base if biglink == "O" else "%s*%s" % (base, biglink)

            return {
                "source_big_node_id": source_sugar_idx,
                "target_big_node_id": target_sugar_idx,
                "big_edge_type": big_edge_type,
                "probability": 1
            }

        def _parse_doubt_target(target_str):
            """解析doubt目标, 如 a? 或 aa?, 返回(target_sugar_idx, target_pos); 解析失败则抛出ValueError"""
            target_match = re.match(r'([a-zA-Z]{1,2})([\d?]+)', target_str)
            if not target_match:
                raise ValueError("Invalid doubt target format: cannot parse '%s'" % target_str)
            target_char = target_match.group(1)
            target_pos_str = target_match.group(2)
            target_pos = int(target_pos_str) if target_pos_str.isdigit() else target_pos_str
            target_sugar_idx = char_to_sugar_idx.get(target_char)
            if target_sugar_idx is None:
                raise ValueError("Invalid doubt target '%s': sugar '%s' not found" % (target_str, target_char))
            return (target_sugar_idx, target_pos)

        def _add_edge_with_reverse(src_idx, tgt_idx, src_pos, tgt_pos, biglink, prob):
            """添加一条大边及其反向边"""
            base = "%s-%s" % (src_pos, tgt_pos)
            big_edge_type = base if biglink == "O" else "%s*%s" % (base, biglink)
            big_edges.append({
                "source_big_node_id": src_idx,
                "target_big_node_id": tgt_idx,
                "big_edge_type": big_edge_type,
                "probability": prob
            })
            # 反向边
            reverse_base = "%s-%s" % (tgt_pos, src_pos)
            reverse_edge_type = reverse_base if biglink == "O" else "%s*%s" % (reverse_base, biglink)
            big_edges.append({
                "source_big_node_id": tgt_idx,
                "target_big_node_id": src_idx,
                "big_edge_type": reverse_edge_type,
                "probability": prob
            })

        # 记录linkage部分的 }*取代基 信息, 用于后续创建小节点和小边
        linkage_doubt_substituents = []

        for linkage in linkages:
            if not linkage:
                continue
            # 统一处理含}的linkage: }标记它前面的|部分为"或"关系
            if '}' in linkage:
                brace_pos = linkage.index('}')
                or_part = linkage[:brace_pos]
                after_brace = linkage[brace_pos + 1:]

                or_options = or_part.split('|')

                # 判断 }*取代基 模式
                if after_brace.startswith('*'):
                    # }*取代基 模式
                    rest = after_brace[1:]
                    sub_end = len(rest)
                    target_part = ""
                    for k, ch in enumerate(rest):
                        if ch == '-' or ch == '{':
                            sub_end = k
                            target_part = rest[k:]
                            break
                    substituent_name = rest[:sub_end]

                    # 解析或选项
                    source_carbon_pairs = []
                    for opt in or_options:
                        info = _parse_doubt_target(opt)
                        source_carbon_pairs.append(info)

                    # 记录这个 }*取代基
                    linkage_doubt_substituents.append({
                        "substituent": substituent_name,
                        "carbon_pairs": source_carbon_pairs
                    })

                    # 如果取代基后面还有连接目标
                    if target_part.startswith('-{'):
                        # 双端或关系
                        target_or_part = target_part[2:]
                        target_or_options = target_or_part.split('|')
                        m_source = len(source_carbon_pairs)
                        n_target = len(target_or_options)
                        verydoubt_prob = 1 / (m_source * n_target) if (m_source * n_target) > 0 else 1
                        target_infos = []
                        for tgt_str in target_or_options:
                            target_info = _parse_doubt_target(tgt_str)
                            target_infos.append(target_info)
                        for source_sugar_idx, source_pos in source_carbon_pairs:
                            for target_sugar_idx, target_pos in target_infos:
                                _add_edge_with_reverse(source_sugar_idx, target_sugar_idx,
                                                       source_pos, target_pos, substituent_name, verydoubt_prob)
                    elif target_part.startswith('-'):
                        # 单端连接
                        target_str = target_part[1:]
                        n_source = len(source_carbon_pairs)
                        doubt_prob = 1 / n_source if n_source > 0 else 1
                        try:
                            firm_edge = _parse_single_linkage(target_str)
                            firm_edge["probability"] = 1
                            big_edges.append(firm_edge)
                            # 反向边(确定边)
                            bet = firm_edge["big_edge_type"]
                            pos_part = bet.split('*')[0] if '*' in bet else bet
                            sp_str = pos_part.split('-')[0]
                            tp_str = pos_part.split('-')[1]
                            src_pos = int(sp_str) if sp_str.isdigit() else sp_str
                            tgt_pos = int(tp_str) if tp_str.isdigit() else tp_str
                            target_sugar_idx = firm_edge["target_big_node_id"]
                            source_sugar_idx_firm = firm_edge["source_big_node_id"]
                            biglink = bet.split('*', 1)[1] if '*' in bet else "O"
                            reverse_base = "%s-%s" % (tgt_pos, src_pos)
                            reverse_edge_type = reverse_base if biglink == "O" else "%s*%s" % (reverse_base, biglink)
                            big_edges.append({
                                "source_big_node_id": target_sugar_idx,
                                "target_big_node_id": source_sugar_idx_firm,
                                "big_edge_type": reverse_edge_type,
                                "probability": 1
                            })
                            target_sugar_idx = firm_edge["target_big_node_id"]
                            pos_part = bet.split('*')[0] if '*' in bet else bet
                            tp_str = pos_part.split('-')[1]
                            target_pos = int(tp_str) if tp_str.isdigit() else tp_str
                        except ValueError:
                            target_info = _parse_doubt_target(target_str)
                            target_sugar_idx, target_pos = target_info
                        # 或关系源端: 每个选项与目标端形成doubt边
                        for source_sugar_idx, source_pos in source_carbon_pairs:
                            _add_edge_with_reverse(source_sugar_idx, target_sugar_idx,
                                                   source_pos, target_pos, substituent_name, doubt_prob)
                    # else: 纯 }*取代基, 无后续连接, 只产生小边, 不产生大边

                elif '-' not in linkage:
                    # 情况1: 纯doubt, 如 a?|b?|c?}
                    n_options = len(or_options)
                    doubt_prob = 1 / n_options if n_options > 0 else 1
                    if or_options and or_options[0]:
                        source_info = _parse_doubt_target(or_options[0])
                        source_sugar_idx, source_pos = source_info
                        for target_str in or_options[1:]:
                            target_info = _parse_doubt_target(target_str)
                            target_sugar_idx, target_pos = target_info
                            _add_edge_with_reverse(source_sugar_idx, target_sugar_idx,
                                                   source_pos, target_pos, "O", doubt_prob)
                elif after_brace.startswith('-{'):
                    # 情况3.5: 双端或关系, 如 c?|a1}-{d2|d4}
                    target_or_part = after_brace[2:]
                    target_or_options = target_or_part.split('|')
                    m_source = len(or_options)
                    n_target = len(target_or_options)
                    verydoubt_prob = 1 / (m_source * n_target) if (m_source * n_target) > 0 else 1
                    source_infos = []
                    for src_str in or_options:
                        source_info = _parse_doubt_target(src_str)
                        source_infos.append(source_info)
                    target_infos = []
                    for tgt_str in target_or_options:
                        target_info = _parse_doubt_target(tgt_str)
                        target_infos.append(target_info)
                    for source_sugar_idx, source_pos in source_infos:
                        for target_sugar_idx, target_pos in target_infos:
                            _add_edge_with_reverse(source_sugar_idx, target_sugar_idx,
                                                   source_pos, target_pos, "O", verydoubt_prob)
                elif after_brace.startswith('-'):
                    # 情况3: -在}后面, 如 a4|b4}-c1
                    n_source = len(or_options)
                    doubt_prob = 1 / n_source if n_source > 0 else 1
                    target_str = after_brace[1:]
                    try:
                        firm_edge = _parse_single_linkage(target_str)
                        firm_edge["probability"] = 1
                        big_edges.append(firm_edge)
                        # 反向边(确定边)
                        bet = firm_edge["big_edge_type"]
                        pos_part = bet.split('*')[0] if '*' in bet else bet
                        sp_str = pos_part.split('-')[0]
                        tp_str = pos_part.split('-')[1]
                        src_pos = int(sp_str) if sp_str.isdigit() else sp_str
                        tgt_pos = int(tp_str) if tp_str.isdigit() else tp_str
                        target_sugar_idx = firm_edge["target_big_node_id"]
                        source_sugar_idx_firm = firm_edge["source_big_node_id"]
                        biglink = bet.split('*', 1)[1] if '*' in bet else "O"
                        reverse_base = "%s-%s" % (tgt_pos, src_pos)
                        reverse_edge_type = reverse_base if biglink == "O" else "%s*%s" % (reverse_base, biglink)
                        big_edges.append({
                            "source_big_node_id": target_sugar_idx,
                            "target_big_node_id": source_sugar_idx_firm,
                            "big_edge_type": reverse_edge_type,
                            "probability": 1
                        })
                        target_sugar_idx = firm_edge["target_big_node_id"]
                        bet = firm_edge["big_edge_type"]
                        pos_part = bet.split('*')[0] if '*' in bet else bet
                        tp_str = pos_part.split('-')[1]
                        target_pos = int(tp_str) if tp_str.isdigit() else tp_str
                        biglink = bet.split('*', 1)[1] if '*' in bet else "O"
                    except ValueError:
                        target_info = _parse_doubt_target(target_str)
                        target_sugar_idx, target_pos = target_info
                        biglink = "O"
                    # 或关系源端: 每个选项与目标端形成doubt边
                    for doubt_str in or_options:
                        source_info = _parse_doubt_target(doubt_str)
                        source_sugar_idx, source_pos = source_info
                        _add_edge_with_reverse(source_sugar_idx, target_sugar_idx,
                                               source_pos, target_pos, biglink, doubt_prob)
                else:
                    # 情况2: -在}前面, 如 b4-d1|a?|e?}
                    dash_pos = or_part.index('-')
                    source_str = or_part[:dash_pos]
                    target_options_str = or_part[dash_pos + 1:]
                    target_options = target_options_str.split('|')
                    n_target = len(target_options)
                    doubt_prob = 1 / n_target if n_target > 0 else 1
                    source_info = _parse_doubt_target(source_str)
                    source_sugar_idx, source_pos = source_info
                    for target_opt in target_options:
                        target_info = _parse_doubt_target(target_opt)
                        target_sugar_idx, target_pos = target_info
                        _add_edge_with_reverse(source_sugar_idx, target_sugar_idx,
                                               source_pos, target_pos, "O", doubt_prob)
                continue

            # 普通确定连接(无}), 如 a3-b1
            edge = _parse_single_linkage(linkage)
            big_edges.append(edge)
            # 反向边
            bet = edge["big_edge_type"]
            pos_part = bet.split('*')[0] if '*' in bet else bet
            sp_str = pos_part.split('-')[0]
            tp_str = pos_part.split('-')[1]
            src_pos = int(sp_str) if sp_str.isdigit() else sp_str
            tgt_pos = int(tp_str) if tp_str.isdigit() else tp_str
            target_sugar_idx = edge["target_big_node_id"]
            source_sugar_idx = edge["source_big_node_id"]
            biglink = bet.split('*', 1)[1] if '*' in bet else "O"
            reverse_base = "%s-%s" % (tgt_pos, src_pos)
            reverse_edge_type = reverse_base if biglink == "O" else "%s*%s" % (reverse_base, biglink)
            big_edges.append({
                "source_big_node_id": target_sugar_idx,
                "target_big_node_id": source_sugar_idx,
                "big_edge_type": reverse_edge_type,
                "probability": 1
            })

        # 处理linkage部分的 }*取代基: 创建小节点和概率小边
        for ld_sub in linkage_doubt_substituents:
            substituent_name = ld_sub["substituent"]
            carbon_pairs = ld_sub["carbon_pairs"]

            # 创建新的小节点
            new_small_node_id = len(small_nodes)
            small_nodes.append({
                "id": new_small_node_id,
                "substituent": substituent_name
            })

            # 确定这个取代基可能连接的所有 (big_node_id, carbon_position) 对
            all_target_pairs = []
            for sugar_idx, pos in carbon_pairs:
                if pos == '?':
                    # 不确定碳位, 连接到该大节点的所有碳位
                    if sugar_idx < len(big_nodes):
                        cc = big_nodes[sugar_idx]["carbon_count_type"]
                        if isinstance(cc, str) and cc.isdigit():
                            cc_int = int(cc)
                            template_idx = indices[sugar_idx] - 1
                            template = parsed_templates[template_idx]
                            firm_non_O_positions = set()
                            for p in range(1, cc_int + 1):
                                if p in template["substituents"]:
                                    firm_non_O_positions.add(p)
                            # 收集被大边占用的位置
                            sugar_occupied = occupied_positions.get(sugar_idx, set())
                            for cp in range(1, cc_int + 1):
                                if cp not in firm_non_O_positions and cp not in sugar_occupied:
                                    all_target_pairs.append((sugar_idx, cp))
                        else:
                            all_target_pairs.append((sugar_idx, '?'))
                else:
                    all_target_pairs.append((sugar_idx, pos))

            n_pairs = len(all_target_pairs)
            prob = 1 / n_pairs if n_pairs > 0 else 1

            # 创建概率小边
            for big_node_id, carbon_pos in all_target_pairs:
                if big_node_id < len(big_nodes) and isinstance(carbon_pos, int):
                    template_idx = indices[big_node_id] - 1
                    template = parsed_templates[template_idx]
                    if carbon_pos <= len(template["stereochemistry"]):
                        stereo_type = template["stereochemistry"][carbon_pos - 1]
                    else:
                        stereo_type = "?"
                else:
                    stereo_type = "?"
                doubt_edge_indices.append(len(small_edges))
                small_edges.append({
                    "big_node_id": big_node_id,
                    "small_node_id": new_small_node_id,
                    "stereo_type": stereo_type,
                    "carbon_position_type": str(carbon_pos),
                    "probability": prob
                })

        result["big_edges"] = big_edges

    # ========== 调整 O 小边的概率，使其与 doubt 边互斥 ==========
        # O边保留 1 - 该位置所有doubt边的概率之和；doubt占满则该位置的O边删除
        small_node_subst = {node["id"]: node["substituent"] for node in small_nodes}
        o_edge_index_by_pos = {}
        for idx, edge in enumerate(small_edges):
            if small_node_subst.get(edge["small_node_id"]) != "O":
                continue
            carbon_pos = edge["carbon_position_type"]
            if not str(carbon_pos).isdigit():
                continue
            if edge.get("probability", 1) == 1:
                key = (edge["big_node_id"], int(carbon_pos))
                o_edge_index_by_pos[key] = idx
        doubt_prob_sum = {}
        for idx in doubt_edge_indices:
            edge = small_edges[idx]
            carbon_pos = edge["carbon_position_type"]
            if not str(carbon_pos).isdigit():
                continue
            key = (edge["big_node_id"], int(carbon_pos))
            doubt_prob_sum[key] = doubt_prob_sum.get(key, 0.0) + edge.get("probability", 1)
        for key, o_idx in o_edge_index_by_pos.items():
            total_doubt = doubt_prob_sum.get(key, 0.0)
            if total_doubt >= 1.0:
                small_edges[o_idx] = None
            elif total_doubt > 0:
                small_edges[o_idx]["probability"] = 1.0 - total_doubt
        small_edges = [e for e in small_edges if e is not None]

        # ========== 删除被掠夺干净后没有任何边的O小节点，并重排小节点ID ==========
        used_small_ids = {e["small_node_id"] for e in small_edges}
        if len(used_small_ids) != len(small_nodes):
            kept_nodes = []
            id_map = {}
            for node in small_nodes:
                if node["id"] in used_small_ids or node["substituent"] != "O":
                    id_map[node["id"]] = len(kept_nodes)
                    kept_nodes.append({"id": len(kept_nodes), "substituent": node["substituent"]})
            for edge in small_edges:
                edge["small_node_id"] = id_map[edge["small_node_id"]]
            small_nodes = kept_nodes
        result["small_nodes"] = small_nodes
        result["small_edges"] = small_edges

        return result


    def print_graph(self, graph_data):
        """打印图结构信息"""
        print("=" * 60)
        print("WURCS Graph Structure")
        print("=" * 60)

        print("\n[Metadata]")
        meta = graph_data["metadata"]
        print("  Templates: %d, Sugars: %d, Linkages: %d, True Linkages: %d" % (
            meta["template_count"], meta["sugar_count"], meta["linkage_count"], meta["true_linkage_count"]
        ))

        print("\n[Big Nodes (Sugar Rings)]")
        for node in graph_data["big_nodes"]:
            print("  Node %d:" % node["id"])
            print("    Carbon Count Type: %s" % node["carbon_count_type"])
            rb = node["ring_bonds"]
            ring_strs = []
            for r in rb:
                rbt = r.get("ring_bond_type", "?")
                pr = r.get("probability", 1)
                ring_strs.append("%s(%s)" % (rbt, pr))
            print("    Ring Bonds: %s" % ", ".join(ring_strs))
            print("    Anomeric Position Type: %s" % node["anomeric_position_type"])
            print("    Anomeric Config Type: %s" % node["anomeric_config_type"])
            if "double_bonds" in node:
                print("    Double Bonds: %s" % ", ".join(node["double_bonds"]))


        print("\n[Small Nodes (Substituents)]")
        for node in graph_data["small_nodes"]:
            print("  Node %d: %s" % (node["id"], node["substituent"]))

        print("\n[Big Edges (Inter-ring Connections)]")
        for edge in graph_data["big_edges"]:
            probability = edge.get("probability", 1)
            bet = edge.get("big_edge_type", "?")
            print("  BigNode%d -> BigNode%d  big_edge_type=%s, probability=%.4f" % (
                edge["source_big_node_id"],
                edge["target_big_node_id"],
                bet,
                probability
            ))

        print("\n[Small Edges (Ring-Substituent)]")
        for edge in graph_data["small_edges"]:
            probability = edge.get("probability", 1)
            print("  BigNode%d -> SmallNode%d: stereo_type=%s, carbon_pos_type=%s, probability=%.4f" % (
                edge["big_node_id"], edge["small_node_id"],
                edge["stereo_type"], edge["carbon_position_type"],
                probability
            ))


if __name__ == "__main__":
    import json
    import sys
    import os
    import csv
    from collections import OrderedDict

    parser = WURCSParser()

    # 基于脚本所在目录的绝对路径，确保输出文件始终保存到代码目录下
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ========== 第一遍：解析所有WURCS，收集vocab ==========
    # 固定使用脚本目录下的 dic.csv
    input_file = os.path.join(SCRIPT_DIR, 'dic.csv')
    if not os.path.isfile(input_file):
        print("Error: dic.csv not found in the script directory!")
        sys.exit(1)
    print("Reading from: %s" % input_file)

    with open(input_file, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    print("Total rows: %d" % len(rows))

    # 检测CSV列名
    if len(rows) > 0:
        columns = list(rows[0].keys())
        print("Detected columns:", columns)
        
        # 确定ID列、标签列和WURCS列（不区分大小写）
        id_column = None
        label_column = None
        wurcs_column = None
        for col in columns:
            col_lower = col.lower().strip()
            if col_lower == 'wurcs':
                wurcs_column = col
            elif col_lower == 'glytoucan id':
                id_column = col
            elif col_lower == 'label':
                label_column = col

        if wurcs_column is None:
            print("Error: No 'WURCS' or 'wurcs' column found in CSV!")
            sys.exit(1)
        if id_column is None:
            print("Error: No 'GlyTouCan ID' column found in CSV!")
            sys.exit(1)
            
        print("Using '%s' as WURCS column, '%s' as ID column, '%s' as label column" % (wurcs_column, id_column, label_column))
    else:
        print("Error: CSV file is empty!")
        sys.exit(1)

    # 收集所有特征值（同时统计频次）
    from collections import Counter
    vocab_sets = {
        "ring_bond_type": set(),
        "substituent": set(),
        "big_edge_type": set(),
        "carbon_count_type": set(),
        "anomeric_position_type": set(),
        "anomeric_config_type": set(),
        "stereo_type": set(),
        "carbon_position_type": set(),
        "double_bonds": set(),
    }
    vocab_counters = {
        "ring_bond_type": Counter(),
        "substituent": Counter(),
        "big_edge_type": Counter(),
        "carbon_count_type": Counter(),
        "anomeric_position_type": Counter(),
        "anomeric_config_type": Counter(),
        "stereo_type": Counter(),
        "carbon_position_type": Counter(),
        "double_bonds": Counter(),
    }

    all_results = []
    success_count = 0
    error_count = 0
    
    # 添加线程锁用于保护共享资源
    import threading
    lock = threading.Lock()

    def process_row(idx, row):
        """处理单行数据的函数，供多线程调用"""
        global success_count, error_count
        
        WURCS = row[wurcs_column]
        glytoucan_id = row[id_column]
        label = row.get(label_column, '') if label_column else ''
        entry = {
            "glytoucan_id": glytoucan_id,
            "label": label,
            "WURCS": WURCS,
            "graph": None
        }
        
        if '~' in WURCS:
            # 包含~的样本直接跳过，不计入error_count
            return entry
        
        try:
            graph = parser.parse(WURCS)
            entry["graph"] = graph
            
            # 收集vocab特征（使用锁保护共享资源）
            with lock:
                success_count += 1
                
                # big_nodes
                for bn in graph["big_nodes"]:
                    vocab_sets["carbon_count_type"].add(bn["carbon_count_type"])
                    vocab_counters["carbon_count_type"][bn["carbon_count_type"]] += 1
                    vocab_sets["anomeric_position_type"].add(bn["anomeric_position_type"])
                    vocab_counters["anomeric_position_type"][bn["anomeric_position_type"]] += 1
                    vocab_sets["anomeric_config_type"].add(bn["anomeric_config_type"])
                    vocab_counters["anomeric_config_type"][bn["anomeric_config_type"]] += 1
                    for rb in bn["ring_bonds"]:
                        rb_prob = rb.get("probability", 1)
                        vocab_sets["ring_bond_type"].add(rb["ring_bond_type"])
                        vocab_counters["ring_bond_type"][rb["ring_bond_type"]] += rb_prob
                    for db in bn.get("double_bonds", []):
                        vocab_sets["double_bonds"].add(db)
                        vocab_counters["double_bonds"][db] += 1

                # small_nodes
                for sn in graph["small_nodes"]:
                    vocab_sets["substituent"].add(sn["substituent"])
                    vocab_counters["substituent"][sn["substituent"]] += 1

                # small_edges
                for se in graph["small_edges"]:
                    se_prob = se.get("probability", 1)
                    vocab_sets["stereo_type"].add(se["stereo_type"])
                    vocab_counters["stereo_type"][se["stereo_type"]] += se_prob
                    vocab_sets["carbon_position_type"].add(se["carbon_position_type"])
                    vocab_counters["carbon_position_type"][se["carbon_position_type"]] += se_prob

                # big_edges
                for be in graph["big_edges"]:
                    prob = be.get("probability", 1)
                    vocab_sets["big_edge_type"].add(be["big_edge_type"])
                    vocab_counters["big_edge_type"][be["big_edge_type"]] += prob

        except Exception as e:
            entry["error"] = str(e)
            with lock:
                error_count += 1
        
        return entry

    # 使用4个线程并行处理
    num_workers = 7
    print(f"Starting parallel parsing with {num_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [executor.submit(process_row, idx, row) for idx, row in enumerate(rows)]
        
        # 收集结果
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            all_results.append(result)
            completed_count += 1
            
            if completed_count % 10000 == 0:
                print(f"Parsed {completed_count} / {len(rows)} ...")

    print("\n=== Parsing Complete ===")
    print("Total rows: %d" % len(rows))
    print("Success: %d" % success_count)
    print("Error: %d" % error_count)
    print("Skipped (contains ~): %d" % (len(rows) - success_count - error_count))

    error_records = [r for r in all_results if r.get("error")]
    if error_records:
        error_csv_path = os.path.join(SCRIPT_DIR, "parse_errors.csv")
        with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["glytoucan_id", "label", "WURCS", "error"])
            for record in error_records:
                writer.writerow([
                    record.get("glytoucan_id", ""),
                    record.get("label", ""),
                    record.get("WURCS", ""),
                    record.get("error", "")
                ])
        print(f"Error records saved to: {error_csv_path}")

    def get_rare_features(counter, min_pct):
        """
        使用百分比阈值判断哪些特征是低频的
        过滤掉出现频率低于 min_pct% 的特征
        
        参数:
            counter: 特征频次计数器
            min_pct: 最小频率阈值（百分比，如1.0表示1%）
        """
        n_features = len(counter)
        if n_features == 0:
            return set()
        
        # 计算该特征类型的总出现频次（所有特征的计数之和）
        total_feature_count = sum(counter.values())
        
        if total_feature_count == 0:
            return set()
        
        # 调试：输出频次最高的几个特征
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        print("    Top 5 most frequent features:")
        for feature, count in sorted_items[:5]:
            freq_pct = count / total_feature_count * 100
            print("      - %s: %.2f (%.1f%%)" % (feature, count, freq_pct))
        
        # 找出频率低于阈值的特征
        rare_features = set()
        print("    Features with frequency < %.2f%%:" % min_pct)
        rare_count = 0
        for feature, count in sorted_items:
            freq_pct = count / total_feature_count * 100
            if freq_pct < min_pct:
                rare_features.add(feature)
                if rare_count < 5:  # 只显示前5个低频特征
                    print("      - %s: %.2f (%.2f%%)" % (feature, count, freq_pct))
                rare_count += 1
        
        if rare_count > 5:
            print("      ... and %d more" % (rare_count - 5))
        
        return rare_features
    
    # ========== 先过滤 true_linkage/(big_edges/2) < 阈值 的样本（不参与迭代）==========
    print("\n=== Pre-Filter: Ratio Filter ===")
    threshold = 0.1
    
    filtered_by_ratio = []
    ratio_filter_count = 0
    
    ratio_sum = 0.0
    ratio_min = float('inf')
    ratio_max = -float('inf')
    ratio_count = 0
    missing_true_linkage = 0
    
    for entry in all_results:
        graph = entry.get("graph")
        if graph is None:
            continue
        
        # 统计包含概率边的 big_edges（所有边都计入，包括有概率的）
        num_big_edges = len(graph["big_edges"])
        
        if "true_linkage_count" not in graph["metadata"]:
            missing_true_linkage += 1
            filtered_by_ratio.append(entry)
            continue
        
        true_linkage = graph["metadata"]["true_linkage_count"]
        
        # 检查 big_edges 是否为 0（避免除以零）
        if num_big_edges == 0:
            if true_linkage == 0:
                # 两者都是0，保留样本
                filtered_by_ratio.append(entry)
            else:
                # big_edges为0但有true_linkage，ratio无穷大，保留
                filtered_by_ratio.append(entry)
            continue
        
        if true_linkage == 0:
            # true_linkage为0，ratio为0，过滤掉
            ratio_filter_count += 1
            continue
        
        # 计算比率: true_linkage / (big_edges/2)
        ratio = true_linkage / (num_big_edges / 2)
        
        ratio_sum += ratio
        ratio_min = min(ratio_min, ratio)
        ratio_max = max(ratio_max, ratio)
        ratio_count += 1
        
        if ratio >= threshold:
            filtered_by_ratio.append(entry)
        else:
            ratio_filter_count += 1
    
    # 输出比率统计
    if ratio_count > 0:
        ratio_avg = ratio_sum / ratio_count
        print("Ratio Statistics:")
        print("  - Min ratio: %.4f" % ratio_min)
        print("  - Max ratio: %.4f" % ratio_max)
        print("  - Avg ratio: %.4f" % ratio_avg)
    if missing_true_linkage > 0:
        print("  - Samples missing true_linkage_count: %d" % missing_true_linkage)
    
    print("Filtered out %d samples with (big_edges/2)/true_linkage < %.2f" % (ratio_filter_count, threshold))
    print("Remaining samples: %d" % len(filtered_by_ratio))
    
    all_results = filtered_by_ratio
    
    # ========== 迭代式百分比过滤 ==========
    min_pct = 0.01  # 频率低于0.01%的特征被认为是低频
    print("\n" + "=" * 60)
    print("Starting Iterative Filtering (Frequency < %.2f%%)" % min_pct)
    print("=" * 60)
    
    iteration = 0
    total_filtered = 0
    current_results = all_results
    
    while True:
        iteration += 1
        print("\n--- Iteration %d ---" % iteration)
        
        # 检查是否还有样本
        if len(current_results) == 0:
            print("  No samples remaining. Stopping.")
            break
        
        # Step 1: 重新统计当前样本的特征频次
        iter_vocab_counters = {
            "substituent": Counter(),
            "stereo_type": Counter(),
            "big_edge_type": Counter(),
            "ring_bond_type": Counter(),
            "carbon_count_type": Counter(),
            "anomeric_position_type": Counter(),
            "anomeric_config_type": Counter(),
            "carbon_position_type": Counter(),
            "double_bonds": Counter(),
        }
        
        for entry in current_results:
            graph = entry.get("graph")
            if graph is None:
                continue
            
            # 统计取代基
            for sn in graph["small_nodes"]:
                prob = sn.get("probability", 1)
                iter_vocab_counters["substituent"][sn["substituent"]] += prob
            
            # 统计stereo_type
            for se in graph["small_edges"]:
                prob = se.get("probability", 1)
                iter_vocab_counters["stereo_type"][se["stereo_type"]] += prob
            
            # 统计big_edge_type
            for be in graph["big_edges"]:
                prob = be.get("probability", 1)
                iter_vocab_counters["big_edge_type"][be["big_edge_type"]] += prob
            
            # 统计ring_bond_type 和 big_node相关特征
            for bn in graph["big_nodes"]:
                # 统计carbon_count_type
                prob = bn.get("probability", 1)
                iter_vocab_counters["carbon_count_type"][bn["carbon_count_type"]] += prob
                
                # 统计anomeric_position_type
                if "anomeric_position_type" in bn:
                    iter_vocab_counters["anomeric_position_type"][bn["anomeric_position_type"]] += prob
                
                # 统计carbon_position_type
                if "carbon_position_type" in bn:
                    iter_vocab_counters["carbon_position_type"][bn["carbon_position_type"]] += prob
                
                # 统计double_bonds
                if "double_bonds" in bn:
                    iter_vocab_counters["double_bonds"][str(bn["double_bonds"])] += prob
                
                # 统计ring_bond_type
                for rb in bn["ring_bonds"]:
                    rb_prob = rb.get("probability", 1)
                    iter_vocab_counters["ring_bond_type"][rb["ring_bond_type"]] += rb_prob
            
            # 统计anomeric_config_type（可能在small_nodes或small_edges中）
            for sn in graph["small_nodes"]:
                if "anomeric_config_type" in sn:
                    prob = sn.get("probability", 1)
                    iter_vocab_counters["anomeric_config_type"][sn["anomeric_config_type"]] += prob
        
        # Step 2: 检测低频特征（频率低于 min_pct%）
        print("  Checking substituents...")
        rare_substituents = get_rare_features(iter_vocab_counters["substituent"], min_pct)
        
        print("  Checking stereo_types...")
        rare_stereo_types = get_rare_features(iter_vocab_counters["stereo_type"], min_pct)
        
        print("  Checking big_edge_types...")
        rare_big_edge_types = get_rare_features(iter_vocab_counters["big_edge_type"], min_pct)
        
        print("  Checking ring_bond_types...")
        rare_ring_bond_types = get_rare_features(iter_vocab_counters["ring_bond_type"], min_pct)
        
        print("  Checking carbon_count_types...")
        rare_carbon_count_types = get_rare_features(iter_vocab_counters["carbon_count_type"], min_pct)
        
        print("  Checking anomeric_position_types...")
        rare_anomeric_position_types = get_rare_features(iter_vocab_counters["anomeric_position_type"], min_pct)
        
        print("  Checking anomeric_config_types...")
        rare_anomeric_config_types = get_rare_features(iter_vocab_counters["anomeric_config_type"], min_pct)
        
        print("  Checking carbon_position_types...")
        rare_carbon_position_types = get_rare_features(iter_vocab_counters["carbon_position_type"], min_pct)
        
        print("  Checking double_bonds...")
        rare_double_bonds = get_rare_features(iter_vocab_counters["double_bonds"], min_pct)
        
        # 统计低频特征信息
        total_rare = len(rare_substituents) + len(rare_stereo_types) + len(rare_big_edge_types) + len(rare_ring_bond_types) + \
                     len(rare_carbon_count_types) + len(rare_anomeric_position_types) + len(rare_anomeric_config_types) + \
                     len(rare_carbon_position_types) + len(rare_double_bonds)
        print("  Detected rare features: %d total" % total_rare)
        print("    - substituents: %d" % len(rare_substituents))
        print("    - stereo_types: %d" % len(rare_stereo_types))
        print("    - big_edge_types: %d" % len(rare_big_edge_types))
        print("    - ring_bond_types: %d" % len(rare_ring_bond_types))
        print("    - carbon_count_types: %d" % len(rare_carbon_count_types))
        print("    - anomeric_position_types: %d" % len(rare_anomeric_position_types))
        print("    - anomeric_config_types: %d" % len(rare_anomeric_config_types))
        print("    - carbon_position_types: %d" % len(rare_carbon_position_types))
        print("    - double_bonds: %d" % len(rare_double_bonds))
        
        # 如果没有低频特征，退出循环
        if total_rare == 0:
            print("  No more rare features detected. Stopping.")
            break
        
        # Step 3: 过滤含有低频特征的样本
        filtered_results = []
        filtered_count = 0
        
        for entry in current_results:
            graph = entry.get("graph")
            if graph is None:
                continue
            
            has_rare_feature = False
            
            # 检查取代基
            for sn in graph["small_nodes"]:
                if sn["substituent"] in rare_substituents:
                    has_rare_feature = True
                    break
            
            if not has_rare_feature:
                # 检查stereo_type
                for se in graph["small_edges"]:
                    if se["stereo_type"] in rare_stereo_types:
                        has_rare_feature = True
                        break
            
            if not has_rare_feature:
                # 检查big_edge_type
                for be in graph["big_edges"]:
                    if be["big_edge_type"] in rare_big_edge_types:
                        has_rare_feature = True
                        break
            
            if not has_rare_feature:
                # 检查big_node相关特征
                for bn in graph["big_nodes"]:
                    # 检查carbon_count_type
                    if bn["carbon_count_type"] in rare_carbon_count_types:
                        has_rare_feature = True
                        break
                    
                    # 检查anomeric_position_type
                    if "anomeric_position_type" in bn and bn["anomeric_position_type"] in rare_anomeric_position_types:
                        has_rare_feature = True
                        break
                    
                    # 检查carbon_position_type
                    if "carbon_position_type" in bn and bn["carbon_position_type"] in rare_carbon_position_types:
                        has_rare_feature = True
                        break
                    
                    # 检查double_bonds
                    if "double_bonds" in bn and str(bn["double_bonds"]) in rare_double_bonds:
                        has_rare_feature = True
                        break
                    
                    # 检查ring_bond_type
                    for rb in bn["ring_bonds"]:
                        if rb["ring_bond_type"] in rare_ring_bond_types:
                            has_rare_feature = True
                            break
                    if has_rare_feature:
                        break
            
            if not has_rare_feature:
                # 检查anomeric_config_type
                for sn in graph["small_nodes"]:
                    if "anomeric_config_type" in sn and sn["anomeric_config_type"] in rare_anomeric_config_types:
                        has_rare_feature = True
                        break
            
            if not has_rare_feature:
                filtered_results.append(entry)
            else:
                filtered_count += 1
        
        print("  Filtered out %d samples" % filtered_count)
        print("  Remaining: %d" % len(filtered_results))
        
        # 调试信息：如果所有样本都被过滤，显示原因
        if len(filtered_results) == 0 and len(current_results) > 0:
            print("  DEBUG: All samples filtered! Checking first few samples...")
            # 检查第一个被过滤的样本包含哪些低频特征
            for i, entry in enumerate(current_results[:3]):
                graph = entry.get("graph")
                if graph is None:
                    continue
                print("    Sample %d:" % (i+1))
                # 检查哪些低频特征导致这个样本被过滤
                for sn in graph["small_nodes"]:
                    if sn["substituent"] in rare_substituents:
                        print("      - Has rare substituent: %s" % sn["substituent"])
                for se in graph["small_edges"]:
                    if se["stereo_type"] in rare_stereo_types:
                        print("      - Has rare stereo_type: %s" % se["stereo_type"])
        
        total_filtered += filtered_count
        current_results = filtered_results
        
        # 如果没有样本被过滤，退出循环
        if filtered_count == 0:
            print("  No samples filtered. Stopping.")
            break
    
    # 最终更新 all_results
    all_results = current_results
    
    print("\n" + "=" * 60)
    print("Iterative Filtering Complete")
    print("  Total iterations: %d" % iteration)
    print("  Total samples filtered: %d" % total_filtered)
    print("  Final remaining samples: %d" % len(all_results))
    print("=" * 60)

    # ========== 基于过滤后的样本重新统计vocab ==========
    filtered_vocab_sets = {
        "ring_bond_type": set(),
        "substituent": set(),
        "big_edge_type": set(),
        "carbon_count_type": set(),
        "anomeric_position_type": set(),
        "anomeric_config_type": set(),
        "stereo_type": set(),
        "carbon_position_type": set(),
        "double_bonds": set(),
    }
    filtered_vocab_counters = {
        "ring_bond_type": Counter(),
        "substituent": Counter(),
        "big_edge_type": Counter(),
        "carbon_count_type": Counter(),
        "anomeric_position_type": Counter(),
        "anomeric_config_type": Counter(),
        "stereo_type": Counter(),
        "carbon_position_type": Counter(),
        "double_bonds": Counter(),
    }
    
    for entry in all_results:
        graph = entry.get("graph")
        if graph is None:
            continue
        
        for bn in graph["big_nodes"]:
            filtered_vocab_sets["carbon_count_type"].add(bn["carbon_count_type"])
            filtered_vocab_counters["carbon_count_type"][bn["carbon_count_type"]] += 1
            filtered_vocab_sets["anomeric_position_type"].add(bn["anomeric_position_type"])
            filtered_vocab_counters["anomeric_position_type"][bn["anomeric_position_type"]] += 1
            filtered_vocab_sets["anomeric_config_type"].add(bn["anomeric_config_type"])
            filtered_vocab_counters["anomeric_config_type"][bn["anomeric_config_type"]] += 1
            for rb in bn["ring_bonds"]:
                rb_prob = rb.get("probability", 1)
                filtered_vocab_sets["ring_bond_type"].add(rb["ring_bond_type"])
                filtered_vocab_counters["ring_bond_type"][rb["ring_bond_type"]] += rb_prob
            for db in bn.get("double_bonds", []):
                filtered_vocab_sets["double_bonds"].add(db)
                filtered_vocab_counters["double_bonds"][db] += 1
        
        for sn in graph["small_nodes"]:
            filtered_vocab_sets["substituent"].add(sn["substituent"])
            filtered_vocab_counters["substituent"][sn["substituent"]] += 1
        
        for se in graph["small_edges"]:
            se_prob = se.get("probability", 1)
            filtered_vocab_sets["stereo_type"].add(se["stereo_type"])
            filtered_vocab_counters["stereo_type"][se["stereo_type"]] += se_prob
            filtered_vocab_sets["carbon_position_type"].add(se["carbon_position_type"])
            filtered_vocab_counters["carbon_position_type"][se["carbon_position_type"]] += se_prob
        
        for be in graph["big_edges"]:
            prob = be.get("probability", 1)
            filtered_vocab_sets["big_edge_type"].add(be["big_edge_type"])
            filtered_vocab_counters["big_edge_type"][be["big_edge_type"]] += prob

    # ========== 生成vocab ==========
    # 对每个特征类别排序后分配索引
    vocab = OrderedDict()
    for key in ["ring_bond_type", "substituent", "big_edge_type",
                "carbon_count_type", "anomeric_position_type",
                "anomeric_config_type", "stereo_type",
                "carbon_position_type", "double_bonds"]:
        sorted_vals = sorted(filtered_vocab_sets[key], key=lambda x: (x != "N/A" and x != "unknown" and x != "none", x))
        vocab[key] = OrderedDict()
        for i, val in enumerate(sorted_vals):
            vocab[key][val] = i
        # 添加MASK token，索引为vocab_size（最后一个）
        vocab[key]["<MASK>"] = len(vocab[key])

    vocab_file = os.path.join(SCRIPT_DIR, 'embedding_vocab.json')
    with open(vocab_file, 'w', encoding='utf-8') as vf:
        json.dump(vocab, vf, ensure_ascii=False, indent=2)
    print("Vocab saved to: %s" % vocab_file)
    for key in vocab:
        print("  %s: %d types" % (key, len(vocab[key])))

    # ========== 输出vocab频次统计txt（基于过滤后的数据）==========
    freq_file = os.path.join(SCRIPT_DIR, 'vocab_freq_stats.txt')
    with open(freq_file, 'w', encoding='utf-8') as ff:
        ff.write("=" * 80 + "\n")
        ff.write("Vocab Frequency Statistics (Filtered)\n")
        ff.write("=" * 80 + "\n\n")
        for key in ["ring_bond_type", "substituent", "big_edge_type",
                    "carbon_count_type", "anomeric_position_type",
                    "anomeric_config_type", "stereo_type",
                    "carbon_position_type", "double_bonds"]:
            ff.write("-" * 60 + "\n")
            ff.write("[%s] (%d unique values)\n" % (key, len(filtered_vocab_counters[key])))
            ff.write("-" * 60 + "\n")
            # 按频次降序排列
            sorted_items = filtered_vocab_counters[key].most_common()
            for val, cnt in sorted_items:
                # 使用浮点数显示，保留2位小数
                ff.write("  %s\t%.2f\n" % (val, cnt))
            ff.write("\n")
    print("Vocab frequency stats saved to: %s" % freq_file)

    # ========== 保存图数据 ==========
    successful_results = [entry for entry in all_results if entry.get("graph") is not None]
    output_json = os.path.join(SCRIPT_DIR, 'output_all.json')
    with open(output_json, 'w', encoding='utf-8') as jf:
        json.dump(successful_results, jf, ensure_ascii=False, indent=2)
    print("Graph data saved to: %s" % output_json)

    # 同时输出可读文本
    output_txt = os.path.join(SCRIPT_DIR, 'output_all.txt')
    original_stdout = sys.stdout
    with open(output_txt, 'w', encoding='utf-8') as fout:
        sys.stdout = fout
        for entry in successful_results:
            print("+" * 60)
            print("GlyTouCan ID: %s | Label: %s" % (entry["glytoucan_id"], entry.get("label", "")))
            print("+" * 60)
            print("Input: %s" % entry["WURCS"])
            parser.print_graph(entry["graph"])
            print()
    sys.stdout = original_stdout
    print("Readable text saved to: %s" % output_txt)

    # ========== 最终汇总统计 ==========
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    print("Original rows: %d" % len(rows))
    print("Parsing success: %d" % success_count)
    print("Parsing errors: %d" % error_count)
    print("Skipped (contains ~): %d" % (len(rows) - success_count - error_count))
    print("Pre-filter (ratio < %.2f): %d" % (threshold, ratio_filter_count))
    print("Iterative filtering iterations: %d" % iteration)
    print("Iterative filtering filtered: %d" % total_filtered)
    print("Final remaining samples: %d" % len(all_results))
    print("=" * 60)