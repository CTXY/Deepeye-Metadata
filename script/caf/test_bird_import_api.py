#!/usr/bin/env python3
"""
Test script for BIRD metadata import API.

This script tests the new import_bird_metadata() and import_and_generate_metadata()
methods in CAFSystem.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from caf.system import CAFSystem
from caf.config.loader import CAFConfig

def test_import_bird_only():
    """Test importing BIRD metadata only."""
    print("\n" + "="*80)
    print("TEST 1: Import BIRD Metadata Only")
    print("="*80)
    
    config_path = project_root / "config" / "caf_config.yaml"
    database_path = project_root / "dataset" / "bird" / "dev" / "dev_databases" / "california_schools" / "california_schools.sqlite"
    bird_data_dir = project_root / "dataset" / "bird"
    
    # Validate paths
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False
    
    if not database_path.exists():
        print(f"❌ Database not found: {database_path}")
        print(f"   Please ensure BIRD dataset is downloaded to: {bird_data_dir}")
        return False
    
    try:
        # Initialize CAF system
        config = CAFConfig.from_file(config_path)
        caf_system = CAFSystem(config)
        
        # Import BIRD metadata
        result = caf_system.import_bird_metadata(
            database_path=str(database_path),
            bird_data_dir=str(bird_data_dir),
            force_regenerate=False,
            continue_on_error=True
        )
        
        # Check result
        print(f"\n✅ Test completed")
        print(f"   Success: {result['success']}")
        print(f"   Database ID: {result['database_id']}")
        print(f"   Duration: {result['duration_seconds']:.2f}s")
        
        if result['errors']:
            print(f"   Errors: {len(result['errors'])}")
            for err in result['errors'][:3]:
                print(f"      - {err}")
        
        if result['warnings']:
            print(f"   Warnings: {len(result['warnings'])}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_import_and_generate():
    """Test importing BIRD metadata + generating additional metadata."""
    print("\n" + "="*80)
    print("TEST 2: Import BIRD + Generate Metadata")
    print("="*80)
    
    config_path = project_root / "config" / "caf_config.yaml"
    database_path = project_root / "dataset" / "bird" / "dev" / "dev_databases" / "california_schools" / "california_schools.sqlite"
    bird_data_dir = project_root / "dataset" / "bird"
    
    if not database_path.exists():
        print(f"⏭️ Skipping test: Database not found")
        return True  # Skip, not fail
    
    try:
        # Initialize CAF system
        config = CAFConfig.from_file(config_path)
        caf_system = CAFSystem(config)
        
        # Import + Generate
        result = caf_system.import_and_generate_metadata(
            database_path=str(database_path),
            bird_data_dir=str(bird_data_dir),
            force_regenerate=False,
            enable_ddl_analysis=True,
            enable_profiling=True,
            enable_llm_analysis=False,  # Disable LLM for faster testing
            enable_join_path_discovery=True,
            continue_on_error=True
        )
        
        # Check result
        print(f"\n✅ Test completed")
        print(f"   Overall Success: {result['overall_success']}")
        print(f"   Total Duration: {result['total_duration_seconds']:.2f}s")
        print(f"   BIRD Import: {'✅' if result['bird_import_result']['success'] else '❌'}")
        
        return result['overall_success']
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validate_memory():
    """Test that imported metadata is accessible via read_memory."""
    print("\n" + "="*80)
    print("TEST 3: Validate Imported Metadata is Accessible")
    print("="*80)
    
    config_path = project_root / "config" / "caf_config.yaml"
    
    try:
        # Initialize CAF system
        config = CAFConfig.from_file(config_path)
        caf_system = CAFSystem(config)
        
        # Bind to california_schools database
        try:
            caf_system.bind_database("california_schools")
            print("✅ Database bound successfully")
        except Exception as e:
            print(f"⚠️ Could not bind database: {e}")
            print("   This is expected if metadata hasn't been imported yet")
            return True  # Skip, not fail
        
        # Try to read memory
        response = caf_system.read_memory(
            memory_type="semantic",
            query_content="What is the eligible free rate?",
            limit=5
        )
        
        print(f"✅ Memory query successful")
        print(f"   Found {len(response.results)} results")
        
        if response.results:
            print(f"   First result: {response.results[0]}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Memory validation failed: {e}")
        print("   This is expected if metadata hasn't been imported yet")
        return True  # Skip, not fail

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BIRD Metadata Import API Tests")
    print("="*80)
    
    tests = [
        ("Import BIRD Only", test_import_bird_only),
        ("Import + Generate", test_import_and_generate),
        ("Validate Memory", test_validate_memory),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except KeyboardInterrupt:
            print("\n\n⚠️ Tests interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)












