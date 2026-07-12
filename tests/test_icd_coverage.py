import sys
from pathlib import Path
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.icd_rag import ICDRetriever


class TestICDCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = ICDRetriever()

    def test_icd_vn_direct_mappings(self):
        """Test L0 direct exact matches and expanded Vietnamese terminology mappings."""
        test_cases = [
            ("ung thư phổi", ["C34", "C34.9"]),
            ("ung thư gan", ["C22", "C22.9"]),
            ("tăng huyết áp", ["I10"]),
            ("suy tim", ["I50", "I50.9"]),
            ("rung nhĩ", ["I48", "I48.9"]),
            ("hội chứng não gan", ["K72.9", "G94"]),
            ("xơ gan do rượu", ["K70.3", "K70"]),
            ("phình động mạch chủ bụng", ["I71.4"]),
            ("nhồi máu cơ tim cũ", ["I25.2"]),
            ("nmct vùng dưới", ["I21.1"]),
            ("block nhánh trái", ["I44.7"]),
            ("suy thận mạn giai đoạn cuối", ["N18.5", "N18.9"]),
            ("vpmpccđ", ["J18", "J18.9"]),
        ]
        for term, expected_prefix in test_cases:
            codes = self.retriever._lookup_single(term, entity_type="CHẨN_ĐOÁN")
            self.assertGreaterEqual(len(codes), 1, f"Expected at least 1 code for '{term}', got {codes}")
            # Check that top returned code matches expected leading prefix or exact code
            top_code = codes[0]
            self.assertTrue(
                any(top_code.startswith(e) or e.startswith(top_code) for e in expected_prefix),
                f"Term '{term}' returned {codes}, expected matching prefix in {expected_prefix}"
            )

    def test_tier_1b_prefix_and_word_containment(self):
        """Test Tier-1b prefix and word containment fallback matches."""
        test_cases = [
            ("nhồi máu cơ tim vùng dưới cũ tiến triển", ["I25.2", "I21", "I21.9"]),
            ("suy tim mạn tính nặng", ["I50", "I50.9"]),
        ]
        for term, expected_prefix in test_cases:
            codes = self.retriever._lookup_single(term, entity_type="CHẨN_ĐOÁN")
            self.assertGreaterEqual(len(codes), 1, f"Expected codes for '{term}', got {codes}")
            top_code = codes[0]
            self.assertTrue(
                any(top_code.startswith(e[:3]) for e in expected_prefix),
                f"Term '{term}' returned {codes}, expected prefix matching {expected_prefix}"
            )

    def test_adaptive_top_k(self):
        """Verify default max_k=1 ensures high precision without false positive expansion."""
        codes_input = ["K70.3", "J18.9", "I10"]
        result = self.retriever._select_adaptive_top_k(codes_input, max_k=1)
        self.assertEqual(result, ["K70.3"], f"Expected ['K70.3'], got {result}")

        codes_input_same_prefix = ["K70.3", "K70.1", "J18.9"]
        result_same = self.retriever._select_adaptive_top_k(codes_input_same_prefix, max_k=2)
        self.assertEqual(result_same, ["K70.3", "K70.1"], f"Expected ['K70.3', 'K70.1'], got {result_same}")


if __name__ == "__main__":
    unittest.main()
