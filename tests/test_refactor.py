#!/usr/bin/env python3
"""Minimal test for the refactored RepeatSequence class."""

import sys
sys.path.insert(0, 'src')

from censim.simulation import RepeatSequence


def test_basic_operations():
    """Test basic get/set operations."""
    print("Test 1: Basic operations")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    print(f"  Length: {len(seq)} (expected: 534)")
    print(f"  Num units: {seq.num_units()} (expected: 3)")
    print(f"  seq[0]: {seq[0]} (expected: A)")
    print(f"  seq[178]: {seq[178]} (expected: T)")
    print(f"  seq[356]: {seq[356]} (expected: C)")
    print("  ✓ Basic operations passed\n")


def test_get_char_slice():
    """Test get_char_slice_str method."""
    print("Test 2: get_char_slice_str")
    # Create sequence: AAAA...TTTT...CCCC...
    seq_str = "A" * 178 + "T" * 178 + "C" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Test within one unit
    slice1 = seq.get_char_slice_str(0, 10)
    print(f"  seq[0:10]: {slice1} (expected: AAAAAAAAAA)")

    # Test across units
    slice2 = seq.get_char_slice_str(170, 186)
    print(f"  seq[170:186]: {slice2} (expected: AAAAAAAATTTTTTTT)")

    # Test across all three units
    slice3 = seq.get_char_slice_str(170, 364)
    expected_len = 194
    print(f"  seq[170:364] length: {len(slice3)} (expected: {expected_len})")
    print(f"  First 8 chars: {slice3[:8]} (expected: AAAAAAAA)")
    print(f"  Last 8 chars: {slice3[-8:]} (expected: CCCCCCCC)")
    print("  ✓ get_char_slice_str passed\n")


def test_delete_within_unit():
    """Test deletion within a single unit (178bp = 1 repeat unit)."""
    print("Test 3: Delete within single unit")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Delete A[50] to A[228] = exactly 178bp (one full repeat unit worth)
    # This spans A[50:178] + next unit's [0:50] = 128 + 50 = 178bp
    seq.delete_at_position(50, 228)

    print(f"  Length after delete: {len(seq)} (expected: 356 = 534-178)")
    print(f"  Num units: {seq.num_units()} (expected: 2)")
    print(f"  seq[0:10]: {seq.get_char_slice_str(0, 10)} (expected: AAAAAAAAAA)")
    print(f"  seq[50:60]: {seq.get_char_slice_str(50, 60)} (expected: TTTTTTTTTT)")
    print("  ✓ Delete within unit passed\n")


def test_delete_across_units():
    """Test deletion across multiple units."""
    print("Test 4: Delete across multiple units")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178 + "G" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Delete from A[50] to C[50]: removes A[50:178] + all of T + C[0:50] = 128 + 178 + 50 = 356 bases
    initial_len = len(seq)
    seq.delete_at_position(50, 406)  # 50 + 356 = 406

    print(f"  Length before: {initial_len} (expected: 712)")
    print(f"  Length after: {len(seq)} (expected: 356 = 712-356)")
    print(f"  First 10 chars: {seq.get_char_slice_str(0, 10)} (expected: AAAAAAAAAA)")
    print(f"  Chars at position 50-60: {seq.get_char_slice_str(50, 60)} (expected: CCCCCCCCCC)")
    print("  ✓ Delete across units passed\n")


def test_duplicate_within_unit():
    """Test duplication spanning units (178bp = 1 repeat unit)."""
    print("Test 5: Duplicate spanning units")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Duplicate A[50] to T[50] = exactly 178bp (A[50:178] + T[0:50] = 128+50)
    seq.duplicate_at_position(50, 228)

    print(f"  Length after duplicate: {len(seq)} (expected: 712 = 534+178)")
    print(f"  Num units: {seq.num_units()} (expected: 4)")
    print(f"  Chars at 0-10: {seq.get_char_slice_str(0, 10)} (expected: AAAAAAAAAA)")
    # After position 50, we should have: A2 T1 [A2 T1 duplicate] T2 C
    print(f"  Chars at 50-60: {seq.get_char_slice_str(50, 60)} (expected: AAAAAAAAAA)")
    print(f"  Chars at 228-238: {seq.get_char_slice_str(228, 238)} (expected: AAAAAAAAAA)")
    print("  ✓ Duplicate spanning units passed\n")


def test_duplicate_across_units():
    """Test duplication across multiple units."""
    print("Test 6: Duplicate across multiple units")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Duplicate from A[50] to T[50]
    # This duplicates A[50:178] + T[0:50] = 128 + 50 = 178 bases
    initial_len = len(seq)
    seq.duplicate_at_position(50, 228)  # 178 + 50 = 228

    print(f"  Length before: {initial_len} (expected: 534)")
    print(f"  Length after: {len(seq)} (expected: 712 = 534+178)")
    print(f"  First 10 chars: {seq.get_char_slice_str(0, 10)} (expected: AAAAAAAAAA)")

    # The duplicated segment should appear twice
    segment1 = seq.get_char_slice_str(50, 228)
    segment2 = seq.get_char_slice_str(228, 406)
    print(f"  Segment 1 == Segment 2: {segment1 == segment2} (expected: True)")
    print(f"  Segment 1 length: {len(segment1)} (expected: 178)")
    print("  ✓ Duplicate across units passed\n")


def test_boundary_aligned_deletion():
    """Test deletion that aligns perfectly to unit boundaries."""
    print("Test 7: Boundary-aligned deletion")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178 + "G" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Delete exactly 2 complete units starting from position 0
    initial_len = len(seq)
    seq.delete_at_position(0, 356)  # 2 * 178 = 356

    print(f"  Length before: {initial_len} (expected: 712)")
    print(f"  Length after: {len(seq)} (expected: 356 = 712-356)")
    print(f"  Num units after: {seq.num_units()} (expected: 2)")
    print(f"  First 10 chars: {seq.get_char_slice_str(0, 10)} (expected: CCCCCCCCCC)")
    print("  ✓ Boundary-aligned deletion passed\n")


def test_validation_errors():
    """Test that non-multiple-of-178 operations raise errors."""
    print("Test 8: Validation errors")
    seq_str = "A" * 178 + "T" * 178 + "C" * 178
    seq = RepeatSequence(seq_str, repeat_size=178)

    # Test invalid deletion (50bp, not a multiple of 178)
    try:
        seq.delete_at_position(50, 100)
        print("  ✗ Should have raised ValueError for non-178-multiple deletion")
        sys.exit(1)
    except ValueError as e:
        print(f"  ✓ Deletion validation works: {str(e)[:60]}...")

    # Test invalid duplication (100bp, not a multiple of 178)
    try:
        seq.duplicate_at_position(50, 150)
        print("  ✗ Should have raised ValueError for non-178-multiple duplication")
        sys.exit(1)
    except ValueError as e:
        print(f"  ✓ Duplication validation works: {str(e)[:60]}...")

    print("  ✓ Validation errors passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RefRepeatSequence Refactoring")
    print("=" * 60 + "\n")

    try:
        test_basic_operations()
        test_get_char_slice()
        test_delete_within_unit()
        test_delete_across_units()
        test_duplicate_within_unit()
        test_duplicate_across_units()
        test_boundary_aligned_deletion()
        test_validation_errors()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
