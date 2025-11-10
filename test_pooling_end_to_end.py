#!/usr/bin/env python3
"""
End-to-end test for all pooling strategies.

This script tests:
1. Direct pooling module imports and usage
2. FT-Transformer with all pooling types
3. CSN-Transformer with all pooling types
4. ModelFactory integration
5. Basic forward pass and gradient computation

Run with: python3 test_pooling_end_to_end.py
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)

    try:
        import torch
        print(f"✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"✗ torch not available: {e}")
        return False

    try:
        from tf_predictor.core.base.pooling import (
            CLSTokenPooling,
            SingleHeadAttentionPooling,
            MultiHeadAttentionPooling,
            WeightedAveragePooling,
            TemporalMultiHeadAttentionPooling,
            create_pooling_module,
            VALID_POOLING_TYPES
        )
        print(f"✓ Pooling module imported")
        print(f"  Valid pooling types: {VALID_POOLING_TYPES}")
    except ImportError as e:
        print(f"✗ Pooling module import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from tf_predictor.core.ft_model import FTTransformerCLSModel
        print(f"✓ FTTransformerCLSModel imported")
    except ImportError as e:
        print(f"✗ FTTransformerCLSModel import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from tf_predictor.core.csn_model import CSNTransformerCLSModel
        print(f"✓ CSNTransformerCLSModel imported")
    except ImportError as e:
        print(f"✗ CSNTransformerCLSModel import failed: {e}")
        traceback.print_exc()
        return False

    try:
        from tf_predictor.core.base.model_factory import ModelFactory
        print(f"✓ ModelFactory imported")
        print(f"  Registered models: {ModelFactory.get_available_models()}")
    except ImportError as e:
        print(f"✗ ModelFactory import failed: {e}")
        traceback.print_exc()
        return False

    print("\n✓ All imports successful!\n")
    return True


def test_pooling_modules():
    """Test direct usage of pooling modules."""
    print("=" * 70)
    print("TEST 2: Pooling Modules")
    print("=" * 70)

    import torch
    from tf_predictor.core.base.pooling import create_pooling_module, VALID_POOLING_TYPES

    batch_size, seq_len, d_token = 4, 10, 64
    n_heads = 8

    x = torch.randn(batch_size, seq_len, d_token)

    results = {}
    for pooling_type in VALID_POOLING_TYPES:
        try:
            pooling = create_pooling_module(
                pooling_type=pooling_type,
                d_token=d_token,
                n_heads=n_heads,
                max_seq_len=seq_len,
                dropout=0.1
            )

            # Forward pass
            output = pooling(x)

            # Count parameters
            num_params = sum(p.numel() for p in pooling.parameters())

            results[pooling_type] = {
                'output_shape': tuple(output.shape),
                'num_params': num_params,
                'success': True
            }

            print(f"✓ {pooling_type:30s} | shape: {tuple(output.shape)} | params: {num_params:6d}")

        except Exception as e:
            results[pooling_type] = {'success': False, 'error': str(e)}
            print(f"✗ {pooling_type:30s} | ERROR: {e}")
            traceback.print_exc()

    # Check all succeeded
    all_success = all(r['success'] for r in results.values())

    if all_success:
        print(f"\n✓ All {len(VALID_POOLING_TYPES)} pooling types working!\n")
    else:
        failed = [k for k, v in results.items() if not v['success']]
        print(f"\n✗ Failed pooling types: {failed}\n")

    return all_success


def test_ft_transformer():
    """Test FT-Transformer with all pooling types."""
    print("=" * 70)
    print("TEST 3: FT-Transformer with All Pooling Types")
    print("=" * 70)

    import torch
    from tf_predictor.core.ft_model import FTTransformerCLSModel
    from tf_predictor.core.base.pooling import VALID_POOLING_TYPES

    batch_size = 4
    seq_len = 10
    num_numerical = 8
    num_categorical = 2
    cat_cardinalities = [100, 5]

    x_num = torch.randn(batch_size, seq_len, num_numerical)
    x_cat = torch.randint(0, 100, (batch_size, num_categorical))
    x_cat[:, 1] = torch.randint(0, 5, (batch_size,))  # Second feature has cardinality 5

    results = {}
    for pooling_type in VALID_POOLING_TYPES:
        try:
            model = FTTransformerCLSModel(
                sequence_length=seq_len,
                num_numerical=num_numerical,
                num_categorical=num_categorical,
                cat_cardinalities=cat_cardinalities,
                output_dim=1,
                d_token=64,
                n_heads=8,
                n_layers=2,
                pooling_type=pooling_type,
                dropout=0.1
            )

            # Forward pass
            output = model(x_num, x_cat)

            # Check gradient flow
            loss = output.sum()
            loss.backward()

            # Get config
            config = model.get_model_config()

            results[pooling_type] = {
                'output_shape': tuple(output.shape),
                'num_params': model.get_num_parameters(),
                'pooling_in_config': config.get('pooling_type') == pooling_type,
                'model_type': config.get('model_type'),
                'success': True
            }

            print(f"✓ {pooling_type:30s} | shape: {tuple(output.shape)} | params: {model.get_num_parameters():8d} | config: {config.get('pooling_type')}")

        except Exception as e:
            results[pooling_type] = {'success': False, 'error': str(e)}
            print(f"✗ {pooling_type:30s} | ERROR: {e}")
            traceback.print_exc()

    # Check all succeeded
    all_success = all(r['success'] for r in results.values())
    all_configs_correct = all(r.get('pooling_in_config', False) for r in results.values() if r['success'])
    all_model_types = set(r.get('model_type') for r in results.values() if r['success'])

    if all_success:
        print(f"\n✓ FT-Transformer works with all {len(VALID_POOLING_TYPES)} pooling types!")
    else:
        failed = [k for k, v in results.items() if not v['success']]
        print(f"\n✗ Failed pooling types: {failed}")

    if all_configs_correct:
        print(f"✓ All configs correctly include pooling_type")
    else:
        print(f"✗ Some configs missing pooling_type")

    if all_model_types == {'ft_transformer'}:
        print(f"✓ Model type is 'ft_transformer' (not 'ft_transformer_cls')")
    else:
        print(f"✗ Unexpected model types: {all_model_types}")

    print()
    return all_success and all_configs_correct


def test_csn_transformer():
    """Test CSN-Transformer with all pooling types."""
    print("=" * 70)
    print("TEST 4: CSN-Transformer with All Pooling Types")
    print("=" * 70)

    import torch
    from tf_predictor.core.csn_model import CSNTransformerCLSModel
    from tf_predictor.core.base.pooling import VALID_POOLING_TYPES

    batch_size = 4
    seq_len = 10
    num_numerical = 8
    num_categorical = 2
    cat_cardinalities = [50, 3]

    x_num = torch.randn(batch_size, seq_len, num_numerical)
    x_cat = torch.randint(0, 50, (batch_size, num_categorical))
    x_cat[:, 1] = torch.randint(0, 3, (batch_size,))  # Second feature has cardinality 3

    results = {}
    for pooling_type in VALID_POOLING_TYPES:
        try:
            model = CSNTransformerCLSModel(
                sequence_length=seq_len,
                num_numerical=num_numerical,
                num_categorical=num_categorical,
                cat_cardinalities=cat_cardinalities,
                output_dim=1,
                d_token=64,
                n_heads=8,
                n_layers=2,
                pooling_type=pooling_type,
                dropout=0.1
            )

            # Forward pass
            output = model(x_num, x_cat)

            # Check gradient flow
            loss = output.sum()
            loss.backward()

            # Get config
            config = model.get_model_config()

            # Check both pathways have pooling
            has_cat_pooling = hasattr(model, 'cat_pooling')
            has_num_pooling = hasattr(model, 'num_pooling')

            results[pooling_type] = {
                'output_shape': tuple(output.shape),
                'num_params': model.get_num_parameters(),
                'pooling_in_config': config.get('pooling_type') == pooling_type,
                'model_type': config.get('model_type'),
                'has_both_poolings': has_cat_pooling and has_num_pooling,
                'success': True
            }

            print(f"✓ {pooling_type:30s} | shape: {tuple(output.shape)} | params: {model.get_num_parameters():8d} | both_poolings: {has_cat_pooling and has_num_pooling}")

        except Exception as e:
            results[pooling_type] = {'success': False, 'error': str(e)}
            print(f"✗ {pooling_type:30s} | ERROR: {e}")
            traceback.print_exc()

    # Check all succeeded
    all_success = all(r['success'] for r in results.values())
    all_have_both = all(r.get('has_both_poolings', False) for r in results.values() if r['success'])
    all_model_types = set(r.get('model_type') for r in results.values() if r['success'])

    if all_success:
        print(f"\n✓ CSN-Transformer works with all {len(VALID_POOLING_TYPES)} pooling types!")
    else:
        failed = [k for k, v in results.items() if not v['success']]
        print(f"\n✗ Failed pooling types: {failed}")

    if all_have_both:
        print(f"✓ All models have both cat_pooling and num_pooling")
    else:
        print(f"✗ Some models missing pooling modules")

    if all_model_types == {'csn_transformer'}:
        print(f"✓ Model type is 'csn_transformer' (not 'csn_transformer_cls')")
    else:
        print(f"✗ Unexpected model types: {all_model_types}")

    print()
    return all_success and all_have_both


def test_model_factory():
    """Test ModelFactory integration."""
    print("=" * 70)
    print("TEST 5: ModelFactory Integration")
    print("=" * 70)

    import torch
    from tf_predictor.core.base.model_factory import ModelFactory, get_default_model_params
    from tf_predictor.core.base.pooling import VALID_POOLING_TYPES

    # Test defaults
    ft_defaults = get_default_model_params('ft_transformer')
    csn_defaults = get_default_model_params('csn_transformer')

    print(f"FT-Transformer defaults: {ft_defaults}")
    print(f"CSN-Transformer defaults: {csn_defaults}")

    if ft_defaults.get('pooling_type') == 'multihead_attention':
        print(f"✓ FT-Transformer default pooling is 'multihead_attention'")
    else:
        print(f"✗ FT-Transformer default pooling is '{ft_defaults.get('pooling_type')}'")
        return False

    if csn_defaults.get('pooling_type') == 'multihead_attention':
        print(f"✓ CSN-Transformer default pooling is 'multihead_attention'")
    else:
        print(f"✗ CSN-Transformer default pooling is '{csn_defaults.get('pooling_type')}'")
        return False

    # Test creating models with different pooling
    print("\nTesting model creation with different pooling types:")

    for pooling_type in ['cls', 'multihead_attention', 'weighted_avg']:
        try:
            model = ModelFactory.create_model(
                model_type='ft_transformer',
                sequence_length=10,
                num_numerical=8,
                num_categorical=0,
                cat_cardinalities=[],
                output_dim=1,
                pooling_type=pooling_type,
                d_token=64,
                n_heads=8,
                n_layers=2
            )

            config = model.get_model_config()
            if config['pooling_type'] == pooling_type:
                print(f"  ✓ Created ft_transformer with pooling_type='{pooling_type}'")
            else:
                print(f"  ✗ Config mismatch: expected '{pooling_type}', got '{config['pooling_type']}'")
                return False

        except Exception as e:
            print(f"  ✗ Failed to create model with pooling_type='{pooling_type}': {e}")
            traceback.print_exc()
            return False

    print("\n✓ ModelFactory integration working!\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("POOLING IMPLEMENTATION - END-TO-END TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Pooling Modules", test_pooling_modules),
        ("FT-Transformer", test_ft_transformer),
        ("CSN-Transformer", test_csn_transformer),
        ("ModelFactory", test_model_factory),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}\n")
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} | {name}")

    all_passed = all(results.values())

    print("=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Module is ready to use!")
        print("=" * 70 + "\n")
        return 0
    else:
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"✗ SOME TESTS FAILED: {', '.join(failed_tests)}")
        print("=" * 70 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
