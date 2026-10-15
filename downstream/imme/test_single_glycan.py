#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小批量测试脚本 - 用于测试单个glycan的WURCS转换和验证流程
"""

import re
import time
import requests

# API端点配置
API_IUPAC2WURCS = "https://api.glycosmos.org/glycanformatconverter/2.7.0/iupaccondensed2wurcs"
API_WURCS2IUPAC = "https://api.glycosmos.org/glycanformatconverter/2.7.0/wurcs2iupaccondensed"
REQUEST_TIMEOUT = 10


def extract_wurcs(response_text):
    """从API响应中提取WURCS字符串"""
    try:
        # 尝试匹配 WURCS= 后面的内容，直到遇到 "、}、空格或换行
        match = re.search(r'WURCS=([^"\'\s\}]+)', response_text)
        if match:
            return f"WURCS={match.group(1)}"
        return ""
    except:
        return ""


def count_replacements(iupac_str):
    """检测需要替换的取代基类型"""
    has_A = False
    has_OMe = False
    has_OP = False
    has_OS = False
    has_P = False
    has_OAc = False
    has_NeuNAc = False
    type_count = 0
    
    # 检查A（糖醛酸）
    sugar_names = [
        'Glc', 'Gal', 'Man', 'Xyl', 'Ara', 'Rha', 'Fuc',
        'Rib', 'Lyx', 'All', 'Alt', 'Gul', 'Ido', 'Tal',
    ]
    for sugar in sugar_names:
        pattern = r'' + sugar + r'A(?=\(|[A-Z\?]|\d|$)'
        if re.search(pattern, iupac_str):
            has_A = True
            type_count += 1
            break
    
    # 检查OMe
    if re.search(r'OMe', iupac_str):
        has_OMe = True
        type_count += 1
    
    # 检查OP（磷酸化）
    has_OP = 'OP' in iupac_str
    if has_OP:
        type_count += 1
    
    # 检查OS（硫酸化）
    has_OS = 'OS' in iupac_str
    if has_OS:
        type_count += 1
    
    # 检查真正的P取代基
    if re.search(r'\d+P\d+', iupac_str) or re.search(r'\d+P\(', iupac_str) or re.search(r'\d+P$', iupac_str):
        has_P = True
        type_count += 1
    
    # 检查OAc
    has_OAc = 'OAc' in iupac_str
    if has_OAc:
        type_count += 1
    
    # 检查NeuNAc（神经氨酸的一种形式，需要转成Neu5NAc）
    has_NeuNAc = 'NeuNAc' in iupac_str
    if has_NeuNAc:
        type_count += 1
    
    return has_A, has_OMe, has_OP, has_OS, has_P, has_OAc, has_NeuNAc, type_count


def replace_OMe_with_placeholder(iupac_str):
    """把OMe替换成?Me作为占位符"""
    return re.sub(r'OMe', '?Me', iupac_str)


def replace_A_with_placeholder(iupac_str):
    """把IUPAC中的大写A替换成?F作为占位符"""
    sugar_names = [
        'Glc', 'Gal', 'Man', 'Xyl', 'Ara', 'Rha', 'Fuc',
        'Rib', 'Lyx', 'All', 'Alt', 'Gul', 'Ido', 'Tal',
    ]
    
    # 情况1: 糖名+A(  -> 糖名+?F(
    pattern1 = r'(' + '|'.join(sugar_names) + r')A\('
    result = re.sub(pattern1, r'\1?F(', iupac_str)
    
    # 情况2: 糖名+A后面跟大写字母或?（修饰） -> 糖名+?F
    pattern2 = r'(' + '|'.join(sugar_names) + r')A(?=[A-Z\?])'
    result = re.sub(pattern2, r'\1?F', result)
    
    # 情况3: 糖名+A在结尾 -> 糖名+?F
    pattern3 = r'(' + '|'.join(sugar_names) + r')A$'
    result = re.sub(pattern3, r'\1?F', result)
    
    # 情况4: 糖名+A+数字（如GlcA2Me） -> 糖名+?F+数字
    pattern4 = r'(' + '|'.join(sugar_names) + r')A(?=\d)'
    result = re.sub(pattern4, r'\1?F', result)
    
    return result


def replace_OP_with_placeholder(iupac_str):
    """把OP替换成?P作为占位符（全局替换）"""
    return iupac_str.replace("OP", "?P")


def replace_OS_with_placeholder(iupac_str):
    """把OS替换成?S作为占位符（全局替换）"""
    return iupac_str.replace("OS", "?S")


def replace_OAc_with_placeholder(iupac_str):
    """把OAc替换成?Ac作为占位符（用于验证）"""
    return iupac_str.replace("OAc", "?Ac")


def replace_NeuNAc(iupac_str):
    """把NeuNAc替换成Neu5Ac"""
    return iupac_str.replace("NeuNAc", "Neu5Ac")




def process_wurcs_remove_f(wurcs, has_A):
    """
    处理WURCS: 
    仅当has_A=True时，删除_?*F这部分，
    对于前面的内容：
    1. 如果有_或-，找第一个分隔符之前的最后一个字母，换成A
    2. 如果没有分隔符，直接把最后一个字母换成A
    """
    if not has_A:
        return wurcs
    
    pattern = r'\[([^\]]*)\]'
    
    def replace_bracket(match):
        content = match.group(1)
        
        while '_?*F' in content:
            f_pos = content.find('_?*F')
            before_f = content[:f_pos]
            after_f = content[f_pos + len('_?*F'):]
            
            sep_match = re.search(r'[_-]', before_f)
            if sep_match:
                prefix = before_f[:sep_match.start()]
                letters = re.findall(r'[a-z]', prefix)
                if letters:
                    last_letter = letters[-1]
                    last_letter_pos = prefix.rfind(last_letter)
                    new_prefix = prefix[:last_letter_pos] + 'A' + prefix[last_letter_pos+1:]
                    rest = before_f[sep_match.start():]
                    content = f'{new_prefix}{rest}{after_f}'
                else:
                    content = f'{before_f}{after_f}'
            else:
                letters = re.findall(r'[a-z]', before_f)
                if letters:
                    last_letter = letters[-1]
                    last_letter_pos = before_f.rfind(last_letter)
                    new_content = before_f[:last_letter_pos] + 'A' + before_f[last_letter_pos+1:]
                    content = f'{new_content}{after_f}'
                else:
                    content = f'{before_f}{after_f}'
        
        return f'[{content}]'
    
    result = re.sub(pattern, replace_bracket, wurcs)
    return result


def process_wurcs_replace_ac_position(wurcs):
    """把WURCS中的 _数字*OCC/3=O 替换成 _?*OCC/3=O"""
    pattern = r'_\d+\*OCC/3=O'
    return re.sub(pattern, '_?*OCC/3=O', wurcs)


def convert_iupac_to_wurcs(iupac_str):
    """调用API把IUPAC转成WURCS"""
    try:
        r = requests.post(API_IUPAC2WURCS, data={"str": iupac_str}, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            wurcs = extract_wurcs(r.text)
            if wurcs:
                return wurcs
            else:
                return f"ERROR: No WURCS found in response. Response: {r.text[:200]}"
        else:
            return f"ERROR: HTTP {r.status_code}. Response: {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


def convert_wurcs_to_iupac(wurcs):
    """调用API把WURCS转成IUPAC Condensed"""
    try:
        import urllib.parse
        import json
        encoded_wurcs = urllib.parse.quote(wurcs, safe='')
        url = f"{API_WURCS2IUPAC}/{encoded_wurcs}"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            # API返回JSON格式，需要提取IUPACcondensed字段
            try:
                data = json.loads(r.text)
                iupac = data.get("IUPACcondensed", "")
                if iupac:
                    return iupac
                # 如果没有IUPACcondensed字段，尝试其他可能的字段名
                for key in ["iupacCondensed", "iupac", "IUPAC"]:
                    if key in data and data[key]:
                        return str(data[key])
                return f"ERROR: No IUPAC found in response. Response: {r.text[:200]}"
            except json.JSONDecodeError:
                # 如果不是JSON，直接返回文本
                return r.text.strip()
        else:
            return f"ERROR: HTTP {r.status_code}. Response: {r.text[:200]}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {str(e)}"


def normalize_naming(iupac_str):
    """
    标准化IUPAC命名，处理GalNAcA这类特殊情况
    例如: GalNAcA -> GalA2NAc
    """
    sugar_names = [
        'Glc', 'Gal', 'Man', 'Xyl', 'Ara', 'Rha', 'Fuc',
        'Rib', 'Lyx', 'All', 'Alt', 'Gul', 'Ido', 'Tal',
    ]
    
    for sugar in sugar_names:
        # 匹配模式: 糖名 + NAc + A + (或结尾)
        pattern = sugar + r'NAcA(\(|$)'
        replacement = sugar + r'A2NAc\1'
        iupac_str = re.sub(pattern, replacement, iupac_str)
    
    return iupac_str


def normalize_iupac_for_comparison(iupac_str):
    """
    标准化IUPAC字符串用于比较
    统计所有字母、数字和问号的频次，完全忽略结构差异
    """
    from collections import Counter
    
    # 保留字母、数字和问号，移除其他特殊字符（如括号、连接符、空格等）
    cleaned = ''.join(c for c in iupac_str if c.isalnum() or c == '?')
    
    # 统计每个字符的频次
    char_counter = Counter(cleaned.upper())  # 统一转大写
    
    # 返回排序后的字符串表示
    normalized = ','.join(f"{char}:{count}" for char, count in sorted(char_counter.items()))
    
    return normalized


def compare_iupac_by_composition(modified_iupac, verified_iupac):
    """
    通过字符频次比较两个IUPAC是否等价
    先进行命名标准化，再比较字符频次
    返回: (is_match, modified_normalized, verified_normalized)
    """
    # 先进行命名标准化（处理GalNAcA -> GalA2NAc这类情况）
    modified_normalized = normalize_naming(modified_iupac)
    verified_normalized = normalize_naming(verified_iupac)
    
    # 然后进行字符频次统计
    from collections import Counter
    
    # 保留字母、数字和问号，移除其他特殊字符
    cleaned_modified = ''.join(c for c in modified_normalized if c.isalnum() or c == '?')
    cleaned_verified = ''.join(c for c in verified_normalized if c.isalnum() or c == '?')
    
    # 统计每个字符的频次
    char_counter_modified = Counter(cleaned_modified.upper())
    char_counter_verified = Counter(cleaned_verified.upper())
    
    # 返回排序后的字符串表示
    modified_norm_str = ','.join(f"{char}:{count}" for char, count in sorted(char_counter_modified.items()))
    verified_norm_str = ','.join(f"{char}:{count}" for char, count in sorted(char_counter_verified.items()))
    
    return char_counter_modified == char_counter_verified, modified_norm_str, verified_norm_str


def test_glycan(original_iupac):
    """测试单个glycan的完整处理流程"""
    print("=" * 80)
    print(f"Testing: {original_iupac}")
    print("=" * 80)
    
    # Step 0: 先标准化命名（处理GalNAcA -> GalA2NAc这类情况）
    # 这一步必须在任何替换之前进行
    normalized_iupac = normalize_naming(original_iupac)
    
    # Step 1: 检测取代基类型（基于标准化后的字符串）
    has_A, has_OMe, has_OP, has_OS, has_P, has_OAc, has_NeuNAc, type_count = count_replacements(normalized_iupac)
    print(f"\n【Step 1】取代基检测:")
    print(f"  A (糖醛酸): {has_A}")
    print(f"  OMe (甲氧基): {has_OMe}")
    print(f"  OP (磷酸化): {has_OP}")
    print(f"  OS (硫酸化): {has_OS}")
    print(f"  P (真实磷酸化): {has_P}")
    print(f"  OAc (乙酰基): {has_OAc}")
    print(f"  NeuNAc (神经氨酸): {has_NeuNAc}")
    print(f"  总类型数: {type_count}")
    
    # A和OP都使用?F占位符，可以共存
    # 两者在WURCS中都生成_?*F，会被统一清理，不会冲突
    # 无需互斥检查
    
    # Step 2: 生成 modified_iupac（用于验证）
    modified_iupac = replace_OMe_with_placeholder(normalized_iupac)
    if has_OAc:
        modified_iupac = replace_OAc_with_placeholder(modified_iupac)
    if has_OP:
        modified_iupac = replace_OP_with_placeholder(modified_iupac)
    if has_OS:
        modified_iupac = replace_OS_with_placeholder(modified_iupac)
    if has_NeuNAc:
        modified_iupac = replace_NeuNAc(modified_iupac)
    
    # 命名标准化（已在开头处理，这里可以省略，但保留以保持一致）
    modified_iupac = normalize_naming(modified_iupac)
    
    print(f"\n【Step 2】modified_iupac (验证目标):")
    print(f"  {modified_iupac}")
    
    # Step 3: 生成 input_iupac（用于API调用）
    input_iupac = normalized_iupac
    input_iupac = replace_OMe_with_placeholder(input_iupac)
    if has_A:
        input_iupac = replace_A_with_placeholder(input_iupac)
    if has_OP:
        input_iupac = replace_OP_with_placeholder(input_iupac)
    if has_OS:
        input_iupac = replace_OS_with_placeholder(input_iupac)
    if has_OAc:
        input_iupac = input_iupac.replace("OAc", "?F")
    if has_NeuNAc:
        input_iupac = replace_NeuNAc(input_iupac)
    
    print(f"\n【Step 3】input_iupac (API输入前):")
    print(f"  {input_iupac}")
    
    # Step 4: API调用
    print(f"\n【Step 4】API调用:")
    print(f"  调用API: {input_iupac}")
    time.sleep(0.05)
    wurcs = convert_iupac_to_wurcs(input_iupac)
    
    if not wurcs or not wurcs.startswith("WURCS="):
        print(f"\n❌ ERROR: WURCS conversion failed")
        print(f"  错误详情: {wurcs}")
        return
    
    print(f"\n【Step 5】WURCS转换结果:")
    print(f"  {wurcs}")
    
    # Step 5: 处理WURCS
    processed_wurcs = process_wurcs_remove_f(wurcs, has_A)
    
    if has_OAc:
        # OAc使用?F占位符，API返回_?*F，需要替换为_?*OCC/3=O以正确映射OAc语义
        processed_wurcs = processed_wurcs.replace('_?*F', '_?*OCC/3=O')
        print(f"\n【Step 6】WURCS处理 (OAc):")
        print(f"  _?*F → _?*OCC/3=O")
    
    print(f"\n【Step 6】处理后WURCS:")
    print(f"  {processed_wurcs}")
    
    # Step 7: 验证
    print(f"\n【Step 7】验证阶段:")
    time.sleep(0.05)
    verified_iupac = convert_wurcs_to_iupac(processed_wurcs)
    print(f"  verified_iupac (原始): {verified_iupac}")
    
    # 命名标准化（处理GalNAcA -> GalA2NAc这类情况）
    verified_iupac_normalized = normalize_naming(verified_iupac)
    print(f"  verified_iupac (标准化后): {verified_iupac_normalized}")
    
    # 验证对比 - 使用组分比较
    match, modified_norm, verified_norm = compare_iupac_by_composition(modified_iupac, verified_iupac_normalized)
    status = "SUCCESS ✅" if match else "MISMATCH ❌"
    
    print(f"\n{'=' * 80}")
    print(f"验证结果: {status}")
    print(f"{'=' * 80}")
    print(f"  modified_iupac:  {modified_iupac}")
    print(f"  verified_iupac:  {verified_iupac_normalized}")
    print(f"  组分比较: modified={modified_norm}, verified={verified_norm}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("小批量测试工具 - Glycan WURCS转换验证")
    print("=" * 80)
    print("\n使用说明:")
    print("1. 直接输入glycan字符串进行测试")
    print("2. 输入 'quit' 或 'exit' 退出程序")
    print("3. 输入空行重新显示说明")
    print("=" * 80 + "\n")
    
    while True:
        glycan = input("请输入glycan (或输入quit退出): ").strip()
        
        if glycan.lower() in ['quit', 'exit', 'q']:
            print("\n再见！👋\n")
            break
        
        if not glycan:
            print("\n" + "=" * 80)
            print("使用说明:")
            print("1. 直接输入glycan字符串进行测试")
            print("2. 输入 'quit' 或 'exit' 退出程序")
            print("3. 输入空行重新显示说明")
            print("=" * 80 + "\n")
            continue
        
        try:
            test_glycan(glycan)
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}\n")
