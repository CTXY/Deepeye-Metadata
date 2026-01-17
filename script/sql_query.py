#!/usr/bin/env python3
"""
SQL查询脚本
允许用户指定数据库路径并运行SQL查询查看结果

使用方法:
1. 修改main()函数中的配置参数
2. 直接运行: python sql_query.py

配置参数说明:
- DB_PATH: 数据库文件路径
- MODE: 运行模式 ("interactive", "query", "tables", "schema", "compare")
- QUERY: 查询模式下的SQL语句
- QUERY1/QUERY2: 比较模式下的两个SQL语句
- TABLE_NAME: 查看表结构模式下的表名
- MAX_ROWS: 最大显示行数
- SHOW_COMPARE_DETAILS: 比较模式是否显示详细信息
"""

import sqlite3
import sys
import os
from typing import List, Any, Optional, Tuple, Dict
from collections import Counter


def connect_to_database(db_path: str) -> sqlite3.Connection:
    """连接到SQLite数据库"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
        return conn
    except sqlite3.Error as e:
        raise sqlite3.Error(f"连接数据库失败: {e}")


def execute_query(conn: sqlite3.Connection, query: str) -> List[sqlite3.Row]:
    """执行SQL查询并返回结果"""
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"执行查询失败: {e}")


def display_results(results: List[sqlite3.Row], max_rows: int = 100):
    """显示查询结果"""
    if not results:
        print("查询结果为空")
        return
    
    # 获取列名
    columns = results[0].keys()
    
    # 计算每列的最大宽度
    col_widths = {}
    for col in columns:
        col_widths[col] = max(len(str(col)), max(len(str(row[col])) for row in results[:max_rows]))
    
    # 打印表头
    header = " | ".join(str(col).ljust(col_widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    
    # 打印数据行
    for i, row in enumerate(results[:max_rows]):
        row_str = " | ".join(str(row[col]).ljust(col_widths[col]) for col in columns)
        print(row_str)
    
    if len(results) > max_rows:
        print(f"\n... 共有 {len(results)} 行数据")


def get_table_info(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """获取数据库中的表信息"""
    query = """
    SELECT name, sql 
    FROM sqlite_master 
    WHERE type='table' 
    ORDER BY name
    """
    return execute_query(conn, query)


def get_table_schema(conn: sqlite3.Connection, table_name: str) -> List[sqlite3.Row]:
    """获取指定表的结构信息"""
    query = f"PRAGMA table_info({table_name})"
    return execute_query(conn, query)


def interactive_mode(conn: sqlite3.Connection):
    """交互式模式"""
    print("进入交互式SQL查询模式")
    print("输入 'help' 查看可用命令，输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nSQL> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'help':
                print_help()
                continue
            elif user_input.lower() == 'tables':
                show_tables(conn)
                continue
            elif user_input.lower().startswith('schema '):
                table_name = user_input[7:].strip()
                show_table_schema(conn, table_name)
                continue
            elif not user_input:
                continue
            
            # 执行SQL查询
            results = execute_query(conn, user_input)
            display_results(results)
            
        except KeyboardInterrupt:
            print("\n\n退出交互式模式")
            break
        except Exception as e:
            print(f"错误: {e}")


def print_help():
    """打印帮助信息"""
    help_text = """
可用命令:
  help          - 显示此帮助信息
  tables        - 显示所有表名
  schema <表名>  - 显示指定表的结构
  quit/exit/q   - 退出程序
  
SQL查询示例:
  SELECT * FROM table_name LIMIT 10;
  SELECT COUNT(*) FROM table_name;
  SELECT column1, column2 FROM table_name WHERE condition;
"""
    print(help_text)


def show_tables(conn: sqlite3.Connection):
    """显示所有表"""
    try:
        tables = get_table_info(conn)
        if tables:
            print("\n数据库中的表:")
            print("-" * 30)
            for table in tables:
                print(f"• {table['name']}")
        else:
            print("数据库中没有表")
    except Exception as e:
        print(f"获取表信息失败: {e}")


def show_table_schema(conn: sqlite3.Connection, table_name: str):
    """显示表结构"""
    try:
        schema = get_table_schema(conn, table_name)
        if schema:
            print(f"\n表 '{table_name}' 的结构:")
            print("-" * 50)
            print(f"{'列名':<20} {'类型':<15} {'非空':<8} {'主键':<8}")
            print("-" * 50)
            for col in schema:
                print(f"{col['name']:<20} {col['type']:<15} {col['notnull']:<8} {col['pk']:<8}")
        else:
            print(f"表 '{table_name}' 不存在")
    except Exception as e:
        print(f"获取表结构失败: {e}")


def row_to_tuple(row: sqlite3.Row) -> Tuple:
    """将Row对象转换为可比较的元组"""
    return tuple(row[col] for col in row.keys())


def sort_key_for_tuple(tup: Tuple) -> Tuple:
    """为元组创建排序键，处理None值"""
    def normalize_for_sort(value: Any) -> Any:
        """将None转换为可比较的值用于排序"""
        if value is None:
            return ('', 0)  # None值排在前面，使用空字符串和0作为标记
        elif isinstance(value, bool):
            return ('', 1 if value else 0)
        elif isinstance(value, (int, float)):
            return ('', value)
        elif isinstance(value, str):
            return (value, 0)
        elif isinstance(value, bytes):
            return (value.decode('utf-8', errors='replace'), 0)
        else:
            return (str(value), 0)
    
    return tuple(normalize_for_sort(v) for v in tup)


def normalize_results(results: List[sqlite3.Row], ignore_column_names: bool = False) -> Tuple[List[Tuple], List[str]]:
    """
    标准化查询结果，返回排序后的元组列表和列名列表
    
    Args:
        results: 查询结果列表
        ignore_column_names: 如果为True，按位置提取数据（忽略列名）；如果为False，按列名提取
    """
    if not results:
        return [], []
    
    if ignore_column_names:
        # 按位置提取数据，忽略列名
        num_cols = len(results[0])
        tuples = [tuple(row[i] for i in range(num_cols)) for row in results]
        columns = [f"col_{i}" for i in range(num_cols)]  # 占位列名
    else:
        # 获取列名（按字母顺序排序以确保一致性）
        columns = sorted(results[0].keys())
        # 转换为元组列表
        tuples = [tuple(row[col] for col in columns) for row in results]
    
    # 使用自定义排序键进行排序，处理None值
    tuples.sort(key=sort_key_for_tuple)
    
    return tuples, columns


def compare_queries(conn: sqlite3.Connection, query1: str, query2: str, 
                   show_details: bool = True) -> bool:
    """
    比较两个SQL查询的结果是否相同
    
    Args:
        conn: 数据库连接
        query1: 第一个SQL查询
        query2: 第二个SQL查询
        show_details: 是否显示详细比较信息
    
    Returns:
        True如果结果相同，False否则
    """
    print("=" * 70)
    print("SQL查询结果比较")
    print("=" * 70)
    
    # 执行两个查询
    print("\n执行查询1...")
    results1 = execute_query(conn, query1)
    print(f"查询1返回 {len(results1)} 行")
    
    print("\n执行查询2...")
    results2 = execute_query(conn, query2)
    print(f"查询2返回 {len(results2)} 行")
    
    # 获取实际列名（用于显示）
    actual_cols1 = list(results1[0].keys()) if results1 else []
    actual_cols2 = list(results2[0].keys()) if results2 else []
    
    # 检查列数
    num_cols1 = len(actual_cols1) if results1 else 0
    num_cols2 = len(actual_cols2) if results2 else 0
    
    if num_cols1 != num_cols2:
        print(f"\n❌ 列数不同: 查询1有 {num_cols1} 列，查询2有 {num_cols2} 列")
        print(f"查询1的列: {actual_cols1}")
        print(f"查询2的列: {actual_cols2}")
        return False
    
    # 标准化结果（忽略列名，按位置比较数据）
    tuples1, _ = normalize_results(results1, ignore_column_names=True)
    tuples2, _ = normalize_results(results2, ignore_column_names=True)
    
    # 比较行数
    if len(tuples1) != len(tuples2):
        print(f"\n❌ 行数不同: 查询1有 {len(tuples1)} 行，查询2有 {len(tuples2)} 行")
        if actual_cols1 != actual_cols2:
            print(f"注意: 列名不同（但不影响数据比较）")
            print(f"查询1的列: {actual_cols1}")
            print(f"查询2的列: {actual_cols2}")
        return False
    
    # 比较数据内容
    if tuples1 == tuples2:
        print(f"\n✅ 查询结果完全相同!")
        print(f"   - 行数: {len(tuples1)}")
        print(f"   - 列数: {num_cols1}")
        if actual_cols1 != actual_cols2:
            print(f"   - 查询1的列名: {', '.join(actual_cols1)}")
            print(f"   - 查询2的列名: {', '.join(actual_cols2)}")
            print(f"   - 注意: 列名不同，但数据内容相同")
        else:
            print(f"   - 列名: {', '.join(actual_cols1)}")
        return True
    
    # 详细比较差异
    print(f"\n❌ 查询结果不同!")
    print(f"   - 行数: 查询1={len(tuples1)}, 查询2={len(tuples2)}")
    print(f"   - 列数: {num_cols1}")
    if actual_cols1 != actual_cols2:
        print(f"   - 查询1的列名: {', '.join(actual_cols1)}")
        print(f"   - 查询2的列名: {', '.join(actual_cols2)}")
        print(f"   - 注意: 列名不同，但已按位置比较数据")
    else:
        print(f"   - 列名: {', '.join(actual_cols1)}")
    
    if show_details:
        # 找出只在查询1中的行
        counter1 = Counter(tuples1)
        counter2 = Counter(tuples2)
        
        only_in_1 = counter1 - counter2
        only_in_2 = counter2 - counter1
        
        # 使用查询1的列名显示（如果列名不同，会同时显示两个查询的列名）
        display_cols = actual_cols1
        
        if only_in_1:
            print(f"\n仅在查询1中的行 (共 {sum(only_in_1.values())} 行):")
            for i, (row, count) in enumerate(list(only_in_1.items())[:10]):
                row_dict = dict(zip(display_cols, row))
                print(f"  {i+1}. {row_dict} (出现 {count} 次)")
            if len(only_in_1) > 10:
                print(f"  ... 还有 {len(only_in_1) - 10} 行未显示")
        
        if only_in_2:
            print(f"\n仅在查询2中的行 (共 {sum(only_in_2.values())} 行):")
            for i, (row, count) in enumerate(list(only_in_2.items())[:10]):
                # 如果列名不同，使用查询2的列名显示
                if actual_cols1 != actual_cols2:
                    row_dict = dict(zip(actual_cols2, row))
                else:
                    row_dict = dict(zip(display_cols, row))
                print(f"  {i+1}. {row_dict} (出现 {count} 次)")
            if len(only_in_2) > 10:
                print(f"  ... 还有 {len(only_in_2) - 10} 行未显示")
        
        # 找出出现次数不同的行
        common_keys = set(counter1.keys()) & set(counter2.keys())
        diff_counts = [(key, counter1[key], counter2[key]) 
                      for key in common_keys 
                      if counter1[key] != counter2[key]]
        
        if diff_counts:
            print(f"\n出现次数不同的行 (共 {len(diff_counts)} 种):")
            for i, (row, count1, count2) in enumerate(diff_counts[:10]):
                row_dict = dict(zip(display_cols, row))
                print(f"  {i+1}. {row_dict}")
                print(f"      查询1中出现 {count1} 次，查询2中出现 {count2} 次")
            if len(diff_counts) > 10:
                print(f"  ... 还有 {len(diff_counts) - 10} 种未显示")
    
    return False


def main():
    # ========== 配置参数 ==========
    # 在这里修改你的配置
    DB_PATH = "/home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/databases/dev_databases/california_schools/california_schools.sqlite"
    
    # 运行模式选择 (选择其中一个)
    MODE = "compare"  # 可选: "interactive", "query", "tables", "schema", "compare"
    
    # 查询模式下的SQL语句
    # 比较两列是否完全相同，输出最终判断结果
    QUERY = '''SELECT T1.School, T1.Street
FROM schools AS T1
INNER
JOIN frpm AS T2
ON T1.CDSCode = T2.CDSCode
WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > 30'''

    # SELECT T1.PostId, T2.Name FROM postHistory AS T1 INNER JOIN badges AS T2 ON T1.UserId = T2.UserId WHERE T1.UserDisplayName = 'Samuel' AND STRFTIME('%Y', T1.CreationDate) = '2013' AND STRFTIME('%Y', T2.Date) = '2013'

    # QUERY = '''SELECT `Percent (%) Eligible Free (K-12)`, CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`
    # FROM frpm AS T2;
    # '''
    
    # 比较模式下的两个SQL语句
    QUERY1 = '''SELECT COUNT(T1.CDSCode) FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`County Name` = 'Fresno' AND T2.NumTstTakr <= 250'''
    
    QUERY2 = '''SELECT COUNT(T1.CDSCode) FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`County` = 'Fresno' AND T2.NumTstTakr <= 250'''
    
    # 查看表结构模式下的表名
    TABLE_NAME = "california_schools"
    
    # 最大显示行数
    MAX_ROWS = 100
    
    # 比较模式是否显示详细信息
    SHOW_COMPARE_DETAILS = True
    # =============================
    
    try:
        # 连接数据库
        conn = connect_to_database(DB_PATH)
        print(f"成功连接到数据库: {DB_PATH}")
        
        # 根据模式执行相应操作
        if MODE == "tables":
            show_tables(conn)
        elif MODE == "schema":
            show_table_schema(conn, TABLE_NAME)
        elif MODE == "query":
            results = execute_query(conn, QUERY)
            display_results(results, MAX_ROWS)
        elif MODE == "compare":
            print("\n查询1:")
            print(QUERY1)
            print("\n查询2:")
            print(QUERY2)
            is_same = compare_queries(conn, QUERY1, QUERY2, SHOW_COMPARE_DETAILS)
            sys.exit(0 if is_same else 1)
        else:  # interactive 或默认
            interactive_mode(conn)
            
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()


if __name__ == "__main__":
    main()
