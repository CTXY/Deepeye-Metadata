#!/usr/bin/env python3
"""
简单的脚本来查看 sub_dev.pkl 中的 database_schema_after_value_retrieval 和 retrieved_values 的内容
"""

import pickle
import json
import sys
from pathlib import Path
from typing import Any, Dict

def to_dict(obj):
    """将对象转换为字典"""
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, 'model_dump'):
        # Pydantic v2
        try:
            return obj.model_dump()
        except:
            pass
    elif hasattr(obj, 'dict'):
        # Pydantic v1
        try:
            return obj.dict()
        except:
            pass
    
    # 处理有__dict__的对象
    if hasattr(obj, '__dict__'):
        result = {}
        obj_dict = vars(obj)
        
        # 处理Pydantic对象的特殊结构
        if '__pydantic_extra__' in obj_dict or '__pydantic_fields_set__' in obj_dict:
            # 这是一个Pydantic对象，尝试从__dict__中提取实际字段
            # Pydantic对象的实际数据可能在嵌套的__dict__中
            if '__dict__' in obj_dict and isinstance(obj_dict['__dict__'], dict):
                result.update(obj_dict['__dict__'])
            # 也尝试直接访问常见属性
            common_attrs = ['question_id', 'question', 'database_id', 'database_path', 
                           'retrieved_values', 'database_schema_after_value_retrieval',
                           'database_schema', 'gold_sql', 'evidence', 'difficulty']
            for attr in common_attrs:
                try:
                    if hasattr(obj, attr):
                        value = getattr(obj, attr)
                        if value is not None:
                            result[attr] = value
                except:
                    pass
        else:
            # 普通对象，直接使用__dict__
            result.update(obj_dict)
        
        # 如果结果为空，尝试通过getattr获取常见属性
        if not result:
            common_attrs = ['question_id', 'question', 'database_id', 'database_path', 
                           'retrieved_values', 'database_schema_after_value_retrieval',
                           'database_schema', 'gold_sql', 'evidence', 'difficulty']
            for attr in common_attrs:
                try:
                    if hasattr(obj, attr):
                        value = getattr(obj, attr)
                        if value is not None:
                            result[attr] = value
                except:
                    pass
        
        return result if result else {'_raw': str(obj)}
    else:
        return {'_raw': str(obj)}

def format_value(value: Any, max_length: int = 200) -> str:
    """格式化值，限制长度"""
    if isinstance(value, (dict, list)):
        try:
            s = json.dumps(value, indent=2, ensure_ascii=False)
            return s
        except:
            s = str(value)
            if len(s) > max_length:
                return s[:max_length] + "..."
            return s
    else:
        s = str(value)
        if len(s) > max_length:
            return s[:max_length] + "..."
        return s

def view_item(item: Any, index: int, show_all: bool = False):
    """查看单个数据项"""
    item_dict = to_dict(item)
    
    print("=" * 100)
    print(f"Item {index}")
    print("=" * 100)
    
    # 基本信息
    print(f"\n基本信息:")
    print(f"  question_id: {item_dict.get('question_id', 'N/A')}")
    print(f"  database_id: {item_dict.get('database_id', 'N/A')}")
    question = item_dict.get('question', 'N/A')
    if isinstance(question, str) and len(question) > 100:
        question = question[:100] + "..."
    print(f"  question: {question}")
    
    # retrieved_values
    print(f"\n{'='*100}")
    print("retrieved_values:")
    print("=" * 100)
    retrieved_values = item_dict.get('retrieved_values')
    if retrieved_values is None:
        print("  (None)")
    elif not retrieved_values:
        print("  (Empty dict)")
    else:
        retrieved_dict = to_dict(retrieved_values)
        for table_name, columns in retrieved_dict.items():
            print(f"\n  表名: {table_name}")
            columns_dict = to_dict(columns)
            if isinstance(columns_dict, dict):
                for column_name, values in columns_dict.items():
                    print(f"    列名: {column_name}")
                    print(f"    值: {format_value(values, max_length=300)}")
            else:
                print(f"    内容: {format_value(columns_dict, max_length=300)}")
    
    # # database_schema_after_value_retrieval
    # print(f"\n{'='*100}")
    # print("database_schema_after_value_retrieval:")
    # print("=" * 100)
    # schema = item_dict.get('database_schema_after_value_retrieval')
    # if schema is None:
    #     print("  (None)")
    # elif not schema:
    #     print("  (Empty dict)")
    # else:
    #     schema_dict = to_dict(schema)
    #     # 显示schema的主要结构
    #     print(f"\n  Schema keys: {list(schema_dict.keys())}")
        
    #     # 如果有tables，显示tables信息
    #     if 'tables' in schema_dict:
    #         tables = schema_dict['tables']
    #         tables_dict = to_dict(tables)
    #         print(f"\n  表数量: {len(tables_dict)}")
    #         print(f"  表名列表: {list(tables_dict.keys())}")
            
    #         # 显示每个表的结构（只显示前几个表，除非show_all=True）
    #         table_list = list(tables_dict.items())

    #         for i, (table_name, table_info) in enumerate(table_list):
    #             print(f"\n  表 {i+1}: {table_name}")
    #             table_info_dict = to_dict(table_info)
    #             print(f"    表信息keys: {list(table_info_dict.keys())}")
                
    #             # 显示列信息
    #             if 'columns' in table_info_dict:
    #                 columns = table_info_dict['columns']
    #                 columns_dict = to_dict(columns)
    #                 print(f"    列数量: {len(columns_dict)}")
    #                 print(f"    列名列表: {list(columns_dict.keys())}")
                    
    #                 # 显示每个列的详细信息（只显示前几个列）
    #                 column_list = list(columns_dict.items())

    #                 for j, (col_name, col_info) in enumerate(column_list):
    #                     print(f"\n      列 {j+1}: {col_name}")
    #                     col_dict = to_dict(col_info)
    #                     for key, value in col_dict.items():
    #                         if key == 'value_examples':
    #                             print(f"        {key}: {format_value(value, max_length=200)}")
    #                         else:
    #                             print(f"        {key}: {format_value(value, max_length=100)}")
                    
        
    #     # 显示其他schema信息
    #     for key, value in schema_dict.items():
    #         if key != 'tables':
    #             print(f"\n  {key}: {format_value(value, max_length=200)}")
    
    # print("\n" + "=" * 100 + "\n")

def main():
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 临时修补pydantic，添加model_validator（如果不存在）
    try:
        import pydantic
        if not hasattr(pydantic, 'model_validator'):
            # 创建一个简单的装饰器来模拟model_validator
            def model_validator(mode="before"):
                def decorator(func):
                    return func
                return decorator
            pydantic.model_validator = model_validator
    except:
        pass
    
    # 文件路径
    pkl_file = project_root / "workspace" / "value_retrieval" / "bird" / "sub_dev.pkl"
    
    if not pkl_file.exists():
        print(f"错误: 文件不存在: {pkl_file}")
        sys.exit(1)
    
    print(f"加载文件: {pkl_file}")
    print("=" * 100)
    
    # 加载pickle文件
    # 使用自定义的Unpickler来避免导入问题
    class SafeUnpickler(pickle.Unpickler):
        def __init__(self, file):
            super().__init__(file)
            self.persistent_load = self._persistent_load
        
        def _persistent_load(self, pid):
            # 处理persistent ID
            raise pickle.UnpicklingError(f"unsupported persistent object: {pid}")
        
        def find_class(self, module, name):
            # 对于app模块的类，创建一个通用的类来存储属性
            if module.startswith('app.'):
                # 创建一个可以存储任意属性的类
                class GenericObject:
                    def __init__(self, *args, **kwargs):
                        # 存储所有参数
                        self._args = args
                        self.__dict__.update(kwargs)
                    
                    def __setstate__(self, state):
                        # 处理pickle的__setstate__
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                        elif isinstance(state, tuple):
                            # 处理tuple格式的状态
                            if len(state) == 2 and isinstance(state[0], dict):
                                self.__dict__.update(state[0])
                            else:
                                self.__dict__ = {'_state': state}
                        else:
                            self.__dict__ = {'_state': state}
                    
                    def __getstate__(self):
                        # 返回当前状态
                        return self.__dict__
                
                return GenericObject
            # 对于其他模块，尝试正常导入
            try:
                return super().find_class(module, name)
            except (ImportError, ModuleNotFoundError, AttributeError, SyntaxError):
                # 如果导入失败，也返回GenericObject
                class GenericObject:
                    def __init__(self, *args, **kwargs):
                        self._args = args
                        self.__dict__.update(kwargs)
                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)
                        else:
                            self.__dict__ = {'_state': state}
                return GenericObject
    
    try:
        with open(pkl_file, 'rb') as f:
            unpickler = SafeUnpickler(f)
            data = unpickler.load()
    except Exception as e:
        print(f"加载pickle文件失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: 如果遇到导入错误，这是正常的，脚本会尝试使用SafeUnpickler处理")
        sys.exit(1)
    
    # 检查data的类型
    # 如果是Dataset对象，提取_data属性
    data_dict = to_dict(data)
    if '_data' in data_dict:
        # 这是一个Dataset对象，提取其中的数据
        data_list = data_dict['_data']
        if not isinstance(data_list, (list, tuple)):
            data_list = list(data_list) if hasattr(data_list, '__iter__') and not isinstance(data_list, (str, bytes)) else [data_list]
    elif isinstance(data, (list, tuple)):
        data_list = list(data)
    elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
        data_list = list(data)
    else:
        # 如果是单个对象，包装成列表
        data_list = [data]
    
    print(f"成功加载，共 {len(data_list)} 个数据项\n")
    
    # 解析命令行参数
    show_all = '--all' in sys.argv or '-a' in sys.argv
    
    # 查找按question_id的参数
    question_id_to_find = None
    for arg in sys.argv[1:]:
        if arg.startswith('--question-id=') or arg.startswith('-q='):
            question_id_to_find = int(arg.split('=')[1])
        elif arg.startswith('--question-id:') or arg.startswith('-q:'):
            question_id_to_find = int(arg.split(':')[1])
        elif arg in ['--question-id', '-q']:
            # 下一个参数应该是question_id
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                question_id_to_find = int(sys.argv[idx + 1])
    
    # 辅助函数：获取question_id
    def get_question_id(item):
        """尝试多种方式获取question_id"""
        # 方法1: 直接访问属性
        if hasattr(item, 'question_id'):
            try:
                val = getattr(item, 'question_id')
                if val is not None and val is not False:
                    return val
            except:
                pass
        
        # 方法2: 通过to_dict
        item_dict = to_dict(item)
        val = item_dict.get('question_id')
        if val is not None and val is not False:
            return val
        
        # 方法3: 尝试访问嵌套的__dict__
        if hasattr(item, '__dict__'):
            obj_dict = vars(item)
            if '__dict__' in obj_dict and isinstance(obj_dict['__dict__'], dict):
                val = obj_dict['__dict__'].get('question_id')
                if val is not None and val is not False:
                    return val
        
        return None
    
    # 如果指定了question_id，查找并显示
    if question_id_to_find is not None:
        found = False
        for i, item in enumerate(data_list):
            item_question_id = get_question_id(item)
            
            if item_question_id == question_id_to_find:
                print(f"找到 question_id={question_id_to_find} 的数据项 (索引: {i})\n")
                view_item(item, i, show_all)
                found = True
                break
        
        if not found:
            print(f"未找到 question_id={question_id_to_find} 的数据项")
            # 显示所有可用的question_id和数据结构信息
            print("\n调试信息 - 第一个数据项的结构:")
            if data_list:
                first_item = data_list[0]
                first_dict = to_dict(first_item)
                print(f"  类型: {type(first_item)}")
                print(f"  字典keys: {list(first_dict.keys())[:20]}")
                if hasattr(first_item, '__dict__'):
                    first_vars = vars(first_item)
                    print(f"  __dict__ keys: {list(first_vars.keys())[:20]}")
                    # 检查嵌套的__dict__
                    if '__dict__' in first_vars and isinstance(first_vars['__dict__'], dict):
                        print(f"  嵌套__dict__ keys: {list(first_vars['__dict__'].keys())[:20]}")
            
            print("\n可用的 question_id 列表 (前20个):")
            question_ids = []
            for i, item in enumerate(data_list[:20]):
                qid = get_question_id(item)
                if qid is not None:
                    question_ids.append((i, qid))
                    print(f"  索引 {i}: question_id={qid}")
            
            if not question_ids:
                print("  (未找到任何question_id)")
        return
    
    # 如果指定了索引，只显示那个索引
    indices = []
    for arg in sys.argv[1:]:
        if arg.isdigit() and not arg.startswith('-'):
            indices.append(int(arg))
        elif arg.startswith('--indices='):
            indices_str = arg.split('=')[1]
            indices = [int(x) for x in indices_str.split(',')]
    
    if indices:
        # 显示指定的索引
        for idx in indices:
            if 0 <= idx < len(data_list):
                view_item(data_list[idx], idx, show_all)
            else:
                print(f"警告: 索引 {idx} 超出范围 (0-{len(data_list)-1})")
    else:
        # 默认显示前3个
        num_items = len(data_list) if show_all else min(3, len(data_list))
        print(f"显示前 {num_items} 个数据项")
        print(f"使用 'python {sys.argv[0]} --question-id=25' 或 'python {sys.argv[0]} -q 25' 查看指定question_id的内容")
        print(f"使用 'python {sys.argv[0]} --all' 查看全部")
        print(f"使用 'python {sys.argv[0]} <index>' 查看指定索引的数据项\n")
        
        for i in range(num_items):
            view_item(data_list[i], i, show_all)
        
        if len(data_list) > num_items:
            print(f"\n提示: 还有 {len(data_list) - num_items} 个数据项未显示")
            print(f"使用 'python {sys.argv[0]} --all' 查看全部")
            print(f"或使用 'python {sys.argv[0]} <index>' 查看指定索引的数据项")

if __name__ == "__main__":
    main()
