"""
测试 Column Retriever 的 Value Match 功能

这个测试脚本展示如何使用新的 value match 功能
"""

def test_basic_value_match_feature(retriever):
    """测试基本的 value match 功能"""
    
    print("=" * 80)
    print("Test 1: Basic Value Match Feature")
    print("=" * 80)
    
    # 测试问题：包含明确的值（California）和实体（schools）
    user_question = "Show me schools in California"
    
    # 调用检索，返回 per_term 结果
    per_term_result, required_tables = retriever.retrieve_anchor_columns(
        user_question=user_question,
        top_k=5,
        return_per_term=True
    )
    
    # 使用内置的打印方法展示结果
    retriever.print_per_term_results(per_term_result)
    
    # 验证结果
    print("\n" + "=" * 80)
    print("Verification:")
    print("=" * 80)
    
    for selection in per_term_result.query_term_selections:
        print(f"\nTerm: '{selection.query_term}'")
        print(f"  - Total candidates: {selection.all_candidates_count}")
        print(f"  - Selected schemas: {len(selection.selected_schemas)}")
        
        # 检查有多少 schema 有 value matches
        schemas_with_values = [
            s for s in selection.selected_schemas_with_values 
            if s.matched_values
        ]
        print(f"  - Schemas with value matches: {len(schemas_with_values)}")
        
        # 显示每个有 value match 的 schema
        for schema_with_val in schemas_with_values:
            print(f"    * {schema_with_val.schema}: {schema_with_val.matched_values}")


def test_formatted_dict_output(retriever):
    """测试字典格式输出"""
    
    print("\n" + "=" * 80)
    print("Test 2: Formatted Dictionary Output")
    print("=" * 80)
    
    user_question = "Find employees in Engineering department with salary greater than 50000"
    
    # 获取结果
    per_term_result, required_tables = retriever.retrieve_anchor_columns(
        user_question=user_question,
        top_k=5,
        return_per_term=True
    )
    
    # 转换为字典格式
    formatted_dict = retriever.format_per_term_results_for_display(per_term_result)
    
    # 打印字典格式
    import json
    print("\nFormatted as dictionary:")
    print(json.dumps(formatted_dict, indent=2, ensure_ascii=False))
    
    # 程序化访问
    print("\n" + "-" * 80)
    print("Programmatic Access:")
    print("-" * 80)
    
    for term, schemas in formatted_dict.items():
        print(f"\nTerm: '{term}'")
        for schema_info in schemas:
            print(f"  Schema: {schema_info['schema']}")
            if schema_info['matched_values']:
                print(f"    ✓ Has value matches: {schema_info['matched_values']}")
                print(f"    ✓ Match types: {schema_info['match_types']}")
                if schema_info['encoding_mappings']:
                    print(f"    ✓ Encoding mappings: {schema_info['encoding_mappings']}")
            else:
                print(f"    ℹ No value matches (semantic/keyword only)")


def test_encoding_mapping_feature(retriever):
    """测试 encoding mapping 功能"""
    
    print("\n" + "=" * 80)
    print("Test 3: Encoding Mapping Feature")
    print("=" * 80)
    
    # 测试包含编码值的查询（如 state 的缩写）
    user_question = "Show me data for CA and NY"
    
    per_term_result, required_tables = retriever.retrieve_anchor_columns(
        user_question=user_question,
        top_k=5,
        return_per_term=True
    )
    
    retriever.print_per_term_results(per_term_result)
    
    # 检查是否有 encoding mappings
    print("\n" + "-" * 80)
    print("Encoding Mappings Found:")
    print("-" * 80)
    
    found_encodings = False
    for selection in per_term_result.query_term_selections:
        for schema_with_val in selection.selected_schemas_with_values:
            if schema_with_val.encoding_mappings:
                found_encodings = True
                print(f"\nSchema: {schema_with_val.schema}")
                print(f"  Matched values: {schema_with_val.matched_values}")
                print(f"  Encoding mappings:")
                for key, value in schema_with_val.encoding_mappings.items():
                    print(f"    {key} -> {value}")
    
    if not found_encodings:
        print("  No encoding mappings found in this query")


def test_semantic_only_match(retriever):
    """测试纯语义匹配（无 value match）的情况"""
    
    print("\n" + "=" * 80)
    print("Test 4: Semantic-Only Match (No Value Match)")
    print("=" * 80)
    
    # 使用抽象的查询，不包含具体值
    user_question = "What is the average enrollment of schools?"
    
    per_term_result, required_tables = retriever.retrieve_anchor_columns(
        user_question=user_question,
        top_k=5,
        return_per_term=True
    )
    
    retriever.print_per_term_results(per_term_result)
    
    # 统计 value match 和 semantic-only match
    print("\n" + "-" * 80)
    print("Match Type Statistics:")
    print("-" * 80)
    
    total_schemas = 0
    value_match_count = 0
    semantic_only_count = 0
    
    for selection in per_term_result.query_term_selections:
        for schema_with_val in selection.selected_schemas_with_values:
            total_schemas += 1
            if schema_with_val.matched_values:
                value_match_count += 1
            else:
                semantic_only_count += 1
    
    print(f"  Total schemas selected: {total_schemas}")
    print(f"  Schemas with value matches: {value_match_count}")
    print(f"  Schemas with semantic-only matches: {semantic_only_count}")


def test_comparison_with_legacy_format(retriever):
    """对比新旧格式的输出"""
    
    print("\n" + "=" * 80)
    print("Test 5: Comparison - Legacy vs New Format")
    print("=" * 80)
    
    user_question = "Show me schools in California"
    
    # 旧格式（return_per_term=False）
    print("\n--- Legacy Format (return_per_term=False) ---")
    selected_columns, required_tables, value_matches = retriever.retrieve_anchor_columns(
        user_question=user_question,
        top_k=5,
        return_per_term=False
    )
    
    print(f"Selected columns: {selected_columns}")
    print(f"Required tables: {required_tables}")
    print(f"Value matches: {value_matches}")
    
    # 新格式（return_per_term=True）
    print("\n--- New Format (return_per_term=True) ---")
    per_term_result, required_tables_new = retriever.retrieve_anchor_columns(
        user_question=user_question,
        top_k=5,
        return_per_term=True
    )
    
    retriever.print_per_term_results(per_term_result)
    
    # 验证一致性
    print("\n" + "-" * 80)
    print("Consistency Check:")
    print("-" * 80)
    
    # 合并后的 schemas 应该与旧格式一致
    merged_schemas_set = set(s.lower() for s in per_term_result.merged_schemas)
    legacy_schemas_set = set(s.lower() for s in selected_columns)
    
    if merged_schemas_set == legacy_schemas_set:
        print("  ✓ Merged schemas match legacy output")
    else:
        print("  ✗ Merged schemas differ from legacy output")
        print(f"    Only in new: {merged_schemas_set - legacy_schemas_set}")
        print(f"    Only in legacy: {legacy_schemas_set - merged_schemas_set}")


def run_all_tests(retriever):
    """运行所有测试"""
    
    print("\n" + "=" * 80)
    print("Running All Tests for Value Match Feature")
    print("=" * 80)
    
    tests = [
        ("Basic Value Match", test_basic_value_match_feature),
        ("Formatted Dict Output", test_formatted_dict_output),
        ("Encoding Mapping", test_encoding_mapping_feature),
        ("Semantic-Only Match", test_semantic_only_match),
        ("Legacy vs New Format", test_comparison_with_legacy_format)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'=' * 80}")
            print(f"Running: {test_name}")
            print(f"{'=' * 80}")
            test_func(retriever)
            print(f"\n✓ {test_name} completed successfully")
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("All Tests Completed")
    print("=" * 80)


# 使用示例
if __name__ == "__main__":
    # 假设已经初始化了 retriever
    # retriever = AnchorColumnRetriever(config)
    # retriever.initialize(llm_client, embedding_client)
    # retriever.build_indexes(database_id, dataframes)
    
    # 运行所有测试
    # run_all_tests(retriever)
    
    # 或者运行单个测试
    # test_basic_value_match_feature(retriever)
    
    print("Test script loaded. Please initialize retriever and call run_all_tests(retriever)")

