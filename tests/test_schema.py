"""领域模型与单位换算的单测。

重点覆盖"假失败"三个来源：单位、精度、写法。
"""

from __future__ import annotations

import pytest

from ni101.schema import (
    GradeUnit,
    MetalUnit,
    OZ_PER_TONNE,
    ResourceClass,
    ResourceRow,
    consistency_deviation,
    expected_metal_tonnes,
    from_tonnes,
    parse_number,
    to_gr_per_tonne,
    to_tonnes,
)


class TestParseNumber:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1,346,100", 1346100.0),
            ("1 234.5", 1234.5),
            ("0.76", 0.76),
            ("(1,234)", -1234.0),
            ("-", None),
            ("n/a", None),
            ("N/A", None),
            ("", None),
            (None, None),
            ("1.18 %", 1.18),
            ("45.8 Mt", 45.8),
            (1782, 1782.0),
            ("\u2013", None),  # en dash
        ],
    )
    def test_parses_common_shapes(self, raw, expected):
        assert parse_number(raw) == expected

    def test_rejects_bool_and_nan(self):
        assert parse_number(True) is None
        assert parse_number(float("nan")) is None


class TestResourceClass:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Indicated", ResourceClass.INDICATED),
            ("Inferred Resources", ResourceClass.INFERRED),
            ("Measured", ResourceClass.MEASURED),
            ("indicated", ResourceClass.INDICATED),
            ("Measured & Indicated", None),  # 合计行不是标准类别
            ("Total", None),
            ("", None),
        ],
    )
    def test_parse(self, raw, expected):
        assert ResourceClass.parse(raw) is expected


class TestMetalUnit:
    def test_koz_before_oz(self):
        assert MetalUnit.parse("Contained Au (koz)") is MetalUnit.KOZ

    def test_mt_and_kt_before_t(self):
        assert MetalUnit.parse("Contained Li2O (Mt)") is MetalUnit.MT
        assert MetalUnit.parse("Ore (kt)") is MetalUnit.KT

    def test_bare_t_only_when_standalone(self):
        assert MetalUnit.parse("Contained Cu (t)") is MetalUnit.T
        # "category" 里也有 t，不能误判成单位 —— 这是真实踩过的坑
        assert MetalUnit.parse("Resource Category") is None
        assert MetalUnit.parse("Metal") is None

    def test_grades_are_not_metal_units(self):
        assert MetalUnit.parse("Grade (g/t Au)") is None


class TestGradeUnit:
    def test_parse(self):
        assert GradeUnit.parse("Grade (g/t Au)") is GradeUnit.G_T
        assert GradeUnit.parse("Cu (%)") is GradeUnit.PERCENT
        assert GradeUnit.parse("Li2O (ppm)") is GradeUnit.PPM
        assert GradeUnit.parse("Resource Category") is None


class TestConversions:
    def test_gram_per_tonne_identity(self):
        assert expected_metal_tonnes(1.0, 1.0) == 1.0
        assert expected_metal_tonnes(45.8, 1.21) == pytest.approx(55.418)

    def test_percent_to_g_t(self):
        assert to_gr_per_tonne(1.0, GradeUnit.PERCENT) == 10000.0
        assert to_gr_per_tonne(0.42, GradeUnit.PERCENT) == pytest.approx(4200.0)

    def test_tonne_ounce_round_trip(self):
        assert from_tonnes(1.0, MetalUnit.OZ) == pytest.approx(OZ_PER_TONNE, rel=1e-9)
        assert from_tonnes(to_tonnes(1782.0, MetalUnit.KOZ), MetalUnit.KOZ) == pytest.approx(1782.0)

    def test_missing_unit_yields_none(self):
        assert to_tonnes(1.0, None) is None
        assert to_gr_per_tonne(1.0, None) is None
        assert expected_metal_tonnes(None, 1.0) is None
        assert expected_metal_tonnes(0.0, 1.0) is None


class TestConsistencyDeviation:
    def test_self_consistent_row(self):
        # 64.3 Mt x 1.18% = 0.75874 Mt，印刷 0.76 Mt
        deviation = consistency_deviation(64.3, 1.18, GradeUnit.PERCENT, 0.76, MetalUnit.MT)
        assert deviation is not None and deviation < 0.02

    def test_contradictory_row(self):
        # 32.7 Mt x 1.05% = 0.34335 Mt，印刷 0.13 Mt -> 偏差约 62%
        deviation = consistency_deviation(32.7, 1.05, GradeUnit.PERCENT, 0.13, MetalUnit.MT)
        assert deviation is not None and deviation > 0.6

    def test_gold_row_in_koz(self):
        # 45.8 Mt @ 1.21 g/t -> 1781.6 koz，印刷 1782 koz
        deviation = consistency_deviation(45.8, 1.21, GradeUnit.G_T, 1782, MetalUnit.KOZ)
        assert deviation is not None and deviation < 0.001

    def test_returns_none_when_field_missing(self):
        assert consistency_deviation(45.8, None, GradeUnit.G_T, 1782, MetalUnit.KOZ) is None
        assert consistency_deviation(45.8, 1.21, GradeUnit.G_T, None, MetalUnit.KOZ) is None


class TestResourceRow:
    def test_coerces_dirty_llm_output(self):
        row = ResourceRow(
            resource_class="Indicated Resources",
            commodity="Au",
            tonnage_mt="45.8",
            grade="1.21",
            grade_unit="g/t",
            contained_metal="1,782",
            contained_metal_unit="koz",
        )
        assert row.resource_class is ResourceClass.INDICATED
        assert row.tonnage_mt == 45.8
        assert row.contained_metal == 1782.0
        assert row.contained_metal_unit is MetalUnit.KOZ
        assert row.is_complete

    def test_derived_values(self):
        row = ResourceRow(
            resource_class="indicated",
            commodity="Au",
            tonnage_mt=45.8,
            grade=1.21,
            grade_unit="g/t",
            contained_metal=1782,
            contained_metal_unit="koz",
        )
        assert row.grade_gr_per_tonne == pytest.approx(1.21)
        assert row.metal_tonnes == pytest.approx(1782 * 1000 / OZ_PER_TONNE)
        assert row.expected_metal_tonnes == pytest.approx(55.418)
        assert row.deviation is not None and row.deviation < 0.001

    def test_identity_key_is_case_insensitive(self):
        row = ResourceRow(resource_class="indicated", commodity="AU")
        assert row.identity_key() == ("indicated", "au")

    def test_incomplete_row_detected(self):
        row = ResourceRow(resource_class="inferred", commodity="Au", tonnage_mt=23.6)
        assert not row.is_complete
