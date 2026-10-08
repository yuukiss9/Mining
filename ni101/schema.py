"""领域模型与单位换算。

设计要点
--------
1. **单位先归一化再比较**。同一份 NI 43-101 里，吨位可能写 Mt / t，品位写 g/t / %，
   金属量写 oz / koz / t / Mt。不归一化就比对，会把正确的抽取判成错的（假失败）。
   所有换算集中在 ``_METAL_TO_TONNE`` / ``to_gr_per_tonne`` 两处，别在别处复制。

2. **三角一致性是本题的硬校验核心**。
   ``吨位(Mt) x 品位(g/t)`` 在数值上恰好等于 ``金属量(t)``：

       1 Mt = 1e6 t；品位 1 g/t 表示每吨矿石含 1 g 金属
       metal(g) = 1e6 * grade  ->  metal(t) = 1e6 * grade / 1e6 = grade

   即 ``expected_metal_tonnes(Mt, g/t) == Mt * g_t``。
   这个恒等式让 CriticMaster 能在**不看原文**的情况下抓出明显错误，
   也是"该 abstain 时必须 abstain"的判据来源。
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------
# 换算常量
# --------------------------------------------------------------------------
OZ_PER_TONNE = 32150.7466          # 1 公吨 = 32150.7466 金衡盎司
GRAMS_PER_TROY_OUNCE = 31.1034768
LB_TO_TONNE = 0.00045359237
PCT_TO_G_T = 10_000.0              # 1 % = 10000 g/t

#: 金属量单位 -> 公吨 的换算因子（唯一真相源，别在别处复制）
_METAL_TO_TONNE: dict[str, float] = {
    "oz": 1.0 / OZ_PER_TONNE,
    "koz": 1_000.0 / OZ_PER_TONNE,
    "t": 1.0,
    "kt": 1_000.0,
    "Mt": 1_000_000.0,
    "lb": LB_TO_TONNE,
}

#: 品位单位 -> g/t 等价 的换算因子
_GRADE_TO_G_T: dict[str, float] = {
    "g/t": 1.0,
    "%": PCT_TO_G_T,
    "ppm": 1.0,
}

# 允许出现的"空值"字面量（NI 43-101 表格里常见）
_NULL_LITERALS = {"", "-", "--", "n/a", "na", "nil", "none", "null", "nd", "notapplicable"}
_MINUS_VARIANTS = {"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2015": "-", "\u00ad": ""}


# --------------------------------------------------------------------------
# 枚举
# --------------------------------------------------------------------------
class ResourceClass(str, Enum):
    """资源量类别。NI 43-101 里常见三类。"""

    MEASURED = "measured"
    INDICATED = "indicated"
    INFERRED = "inferred"

    @classmethod
    def parse(cls, raw: Any) -> Optional["ResourceClass"]:
        """从自由文本里识别类别（LLM 输出常带 'Indicated Resources' 这类后缀）。"""
        if raw is None:
            return None
        if isinstance(raw, ResourceClass):
            return raw
        text = str(raw).strip().lower()
        if not text:
            return None
        # "Measured & Indicated" 是合计行，必须早于单词匹配判断，否则会误判成 measured
        if "measured&indicated" in text.replace(" ", "") or "m&i" in text.replace(" ", ""):
            return None  # 合计行不是标准类别，交由 Critic 判定为异常行
        for member in (cls.MEASURED, cls.INDICATED, cls.INFERRED):
            if member.value in text:
                return member
        return None


class GradeUnit(str, Enum):
    G_T = "g/t"
    PERCENT = "%"
    PPM = "ppm"

    @classmethod
    def parse(cls, raw: Any) -> Optional["GradeUnit"]:
        if raw is None:
            return None
        if isinstance(raw, GradeUnit):
            return raw
        text = str(raw).strip().lower().replace(" ", "")
        if not text:
            return None
        if "g/t" in text or "gpt" in text or "grampertonne" in text:
            return cls.G_T
        if "%" in text or "percent" in text or "pct" in text:
            return cls.PERCENT
        if "ppm" in text:
            return cls.PPM
        return None


class MetalUnit(str, Enum):
    OZ = "oz"
    KOZ = "koz"
    T = "t"
    KT = "kt"
    MT = "Mt"
    LB = "lb"

    @classmethod
    def parse(cls, raw: Any) -> Optional["MetalUnit"]:
        if raw is None:
            return None
        if isinstance(raw, MetalUnit):
            return raw
        compact = str(raw).strip().lower().replace(" ", "").replace(",", "")
        if not compact:
            return None
        # 顺序重要：koz 必须早于 oz，kt/Mt 必须早于 t
        if "koz" in compact or "000oz" in compact:
            return cls.KOZ
        if "moz" in compact:
            return cls.KOZ  # 数量级差 1000，交给量级校验兜住
        if re.search(r"(?:^|[^a-z])oz(?:[^a-z]|$)", compact):
            return cls.OZ
        if "kt" in compact:
            return cls.KT
        if "mt" in compact:
            return cls.MT
        if "lb" in compact or "pound" in compact:
            return cls.LB
        # 单独的 "t" 才算吨，避免把 "category" 里的 t 误判成单位
        if re.search(r"(?:^|[^a-z])t(?:[^a-z]|$)", compact) or "tonne" in compact:
            return cls.T
        return None


# --------------------------------------------------------------------------
# 数值解析
# --------------------------------------------------------------------------
def parse_number(raw: Any) -> Optional[float]:
    """把 LLM / 表格单元格里的数值文本转成 float。

    处理 ``"1,234.5"`` / ``"1 234"`` / ``"(1,234)"``（会计负数）/ ``"n/a"`` 等形态。
    解析不出来返回 None，由 Critic 记为缺失字段——**不抛异常**，
    因为真实 PDF 里本来就有 "-" 和 "n/a"。
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return None if math.isnan(value) or math.isinf(value) else value
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    for bad, good in _MINUS_VARIANTS.items():
        text = text.replace(bad, good)
    if text.lower().replace(" ", "") in _NULL_LITERALS:
        return None

    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]

    text = text.replace(",", "").replace(" ", "").replace("\u00a0", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return -value if negative else value


# --------------------------------------------------------------------------
# 归一化换算（唯一实现处）
# --------------------------------------------------------------------------
def to_tonnes(value: Optional[float], unit: Optional[MetalUnit]) -> Optional[float]:
    """金属量 -> 公吨。"""
    if value is None or unit is None:
        return None
    return value * _METAL_TO_TONNE[unit.value]


def from_tonnes(tonnes: Optional[float], unit: MetalUnit) -> Optional[float]:
    """公吨 -> 指定单位（换算的逆运算，报告展示用）。"""
    if tonnes is None:
        return None
    return tonnes / _METAL_TO_TONNE[unit.value]


def to_gr_per_tonne(value: Optional[float], unit: Optional[GradeUnit]) -> Optional[float]:
    """品位 -> g/t 等价。"""
    if value is None or unit is None:
        return None
    return value * _GRADE_TO_G_T[unit.value]


def expected_metal_tonnes(
    tonnage_mt: Optional[float], grade_gr_per_tonne: Optional[float]
) -> Optional[float]:
    """三角一致性公式：``金属量(t) == 吨位(Mt) x 品位(g/t)``。

    推导：吨位 Mt -> 吨 x1e6；品位 g/t -> 每吨含 g 数；
    金属克数 = Mt*1e6*grade；换算回吨需 /1e6，故数值上恰好等于 Mt*grade。
    """
    if tonnage_mt is None or grade_gr_per_tonne is None:
        return None
    if tonnage_mt <= 0 or grade_gr_per_tonne <= 0:
        return None
    return tonnage_mt * grade_gr_per_tonne


def consistency_deviation(
    tonnage_mt: Optional[float],
    grade: Optional[float],
    grade_unit: Optional[GradeUnit],
    contained_metal: Optional[float],
    metal_unit: Optional[MetalUnit],
) -> Optional[float]:
    """返回三角一致性的**相对偏差**（0.15 表示 15%）。

    任一字段缺失时返回 None —— 缺失由 Critic 另按"字段缺失"处理，
    不能和"数值矛盾"混为一谈，否则会把正常行误判成异常。
    """
    gr = to_gr_per_tonne(grade, grade_unit)
    metal_t = to_tonnes(contained_metal, metal_unit)
    expect_t = expected_metal_tonnes(tonnage_mt, gr)
    if expect_t is None or metal_t is None or metal_t <= 0:
        return None
    return abs(metal_t - expect_t) / expect_t


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class ResourceRow(BaseModel):
    """一个资源量数据行 —— 抽取的最小单元。"""

    model_config = ConfigDict(protected_namespaces=())

    resource_class: Optional[ResourceClass] = None
    commodity: str = ""
    tonnage_mt: Optional[float] = None
    grade: Optional[float] = None
    grade_unit: Optional[GradeUnit] = None
    contained_metal: Optional[float] = None
    contained_metal_unit: Optional[MetalUnit] = None
    source_page: Optional[int] = None
    source_snippet: Optional[str] = None
    confidence: Optional[float] = None

    # 行级拒答：系统判断这一行无法可靠给出时使用
    abstain: bool = False
    abstain_reason: Optional[str] = None

    @field_validator("tonnage_mt", "grade", "contained_metal", mode="before")
    @classmethod
    def _coerce_number(cls, value: Any) -> Optional[float]:
        return parse_number(value)

    @field_validator("resource_class", mode="before")
    @classmethod
    def _coerce_class(cls, value: Any) -> Any:
        if value is None or isinstance(value, ResourceClass):
            return value
        return ResourceClass.parse(value)

    @field_validator("grade_unit", mode="before")
    @classmethod
    def _coerce_grade_unit(cls, value: Any) -> Any:
        if value is None or isinstance(value, GradeUnit):
            return value
        return GradeUnit.parse(value)

    @field_validator("contained_metal_unit", mode="before")
    @classmethod
    def _coerce_metal_unit(cls, value: Any) -> Any:
        if value is None or isinstance(value, MetalUnit):
            return value
        return MetalUnit.parse(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> Optional[float]:
        parsed = parse_number(value)
        if parsed is None:
            return None
        return max(0.0, min(1.0, parsed))

    # -- 派生量 ----------------------------------------------------------
    @property
    def grade_gr_per_tonne(self) -> Optional[float]:
        return to_gr_per_tonne(self.grade, self.grade_unit)

    @property
    def metal_tonnes(self) -> Optional[float]:
        return to_tonnes(self.contained_metal, self.contained_metal_unit)

    @property
    def expected_metal_tonnes(self) -> Optional[float]:
        return expected_metal_tonnes(self.tonnage_mt, self.grade_gr_per_tonne)

    @property
    def deviation(self) -> Optional[float]:
        return consistency_deviation(
            self.tonnage_mt,
            self.grade,
            self.grade_unit,
            self.contained_metal,
            self.contained_metal_unit,
        )

    @property
    def is_complete(self) -> bool:
        return all(
            v is not None
            for v in (
                self.tonnage_mt,
                self.grade,
                self.grade_unit,
                self.contained_metal,
                self.contained_metal_unit,
            )
        )

    def identity_key(self) -> tuple[str, str]:
        """用于与 ground truth 对齐的键：(资源类别, 矿种)。"""
        cls = self.resource_class.value if self.resource_class else "unknown"
        return cls, (self.commodity or "unknown").strip().lower()

    def summary(self) -> str:
        if self.abstain:
            return f"[abstain] {self.identity_key()} :: {self.abstain_reason}"
        grade_unit = self.grade_unit.value if self.grade_unit else "?"
        metal_unit = self.contained_metal_unit.value if self.contained_metal_unit else "?"
        return (
            f"{self.identity_key()} :: {self.tonnage_mt} Mt "
            f"@ {self.grade} {grade_unit} => {self.contained_metal} {metal_unit}"
        )


class ExtractionResult(BaseModel):
    """Extractor 一次抽取的产物。"""

    model_config = ConfigDict(protected_namespaces=())

    rows: list[ResourceRow] = Field(default_factory=list)
    source: str = "unknown"          # llm | rule
    note: str = ""
    abstain: bool = False
    abstain_reasons: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    def by_class(self, resource_class: ResourceClass) -> list[ResourceRow]:
        return [r for r in self.rows if r.resource_class == resource_class]


class GroundTruthRow(BaseModel):
    """ground truth 单行。

    ``expect_abstain`` 是本项目评分协议的核心标记：原文数值自相矛盾时，
    正确行为是拒答而不是硬给。
    """

    model_config = ConfigDict(protected_namespaces=())

    resource_class: Optional[ResourceClass] = None
    commodity: str = ""
    tonnage_mt: Optional[float] = None
    grade: Optional[float] = None
    grade_unit: Optional[GradeUnit] = None
    contained_metal: Optional[float] = None
    contained_metal_unit: Optional[MetalUnit] = None
    source_page: Optional[int] = None

    expect_abstain: bool = False
    note: str = ""

    @field_validator("tonnage_mt", "grade", "contained_metal", mode="before")
    @classmethod
    def _coerce_number(cls, value: Any) -> Optional[float]:
        return parse_number(value)

    @field_validator("resource_class", mode="before")
    @classmethod
    def _coerce_class(cls, value: Any) -> Any:
        if value is None or isinstance(value, ResourceClass):
            return value
        return ResourceClass.parse(value)

    @field_validator("grade_unit", mode="before")
    @classmethod
    def _coerce_grade_unit(cls, value: Any) -> Any:
        if value is None or isinstance(value, GradeUnit):
            return value
        return GradeUnit.parse(value)

    @field_validator("contained_metal_unit", mode="before")
    @classmethod
    def _coerce_metal_unit(cls, value: Any) -> Any:
        if value is None or isinstance(value, MetalUnit):
            return value
        return MetalUnit.parse(value)

    def identity_key(self) -> tuple[str, str]:
        cls = self.resource_class.value if self.resource_class else "unknown"
        return cls, (self.commodity or "unknown").strip().lower()


class GroundTruthDocument(BaseModel):
    """一份 PDF 对应的 ground truth 文件。"""

    model_config = ConfigDict(protected_namespaces=())

    pdf: str = ""
    company: str = ""
    project: str = ""
    reported_date: str = ""
    cut_off_grade: str = ""
    rows: list[GroundTruthRow] = Field(default_factory=list)
