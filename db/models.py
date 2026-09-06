"""
SQLAlchemy models for the supply chain detector.

Tables:
  - Package       : one row per unique package seen across registries
  - ScanResult    : one scan row per package (updated on re-scan)
  - DetectorSignal: individual signal scores that feed the final risk score
  - Allowlist     : manually approved packages (never flagged again)
"""

from datetime import datetime, timezone
import enum

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, Text, ForeignKey, Enum
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


def _utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ── Enums ──────────────────────────────────────────────────────────────────

class Registry(str, enum.Enum):
    NPM   = "npm"
    PYPI  = "pypi"

class RiskLevel(str, enum.Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    UNKNOWN  = "unknown"


# ── Base ───────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Models ─────────────────────────────────────────────────────────────────

class Package(Base):
    __tablename__ = "packages"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    registry        = Column(Enum(Registry, native_enum=False), nullable=False, index=True)
    name            = Column(String(256), nullable=False, index=True)
    version         = Column(String(64))
    author          = Column(String(256))
    description     = Column(Text)
    homepage        = Column(String(512))
    repository_url  = Column(String(512))

    publisher_email             = Column(String(256))
    publisher_created           = Column(DateTime)
    total_packages_by_publisher = Column(Integer, default=0)

    weekly_downloads  = Column(Integer, default=0)
    github_stars      = Column(Integer, default=0)
    first_published   = Column(DateTime)
    last_updated      = Column(DateTime)

    discovered_at = Column(DateTime, default=_utc_now)
    raw_metadata  = Column(Text)

    scan_result = relationship("ScanResult", back_populates="package",
                               uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        reg_val = self.registry.value if hasattr(self.registry, 'value') else self.registry
        return f"<Package {reg_val}/{self.name}>"


class ScanResult(Base):
    __tablename__ = "scan_results"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    package_id = Column(Integer, ForeignKey("packages.id"), unique=True, nullable=False)

    risk_score = Column(Float, default=0.0)
    risk_level = Column(Enum(RiskLevel, native_enum=False), default=RiskLevel.UNKNOWN)

    is_typosquat            = Column(Boolean, default=False)
    is_dependency_confusion = Column(Boolean, default=False)
    has_suspicious_scripts  = Column(Boolean, default=False)

    closest_match      = Column(String(256))
    closest_match_dist = Column(Integer)

    verdict_reason = Column(Text)
    scanned_at     = Column(DateTime, default=_utc_now)

    package = relationship("Package", back_populates="scan_result")
    signals = relationship("DetectorSignal", back_populates="scan_result",
                           cascade="all, delete-orphan")

    def __repr__(self):
        level_val = self.risk_level.value if hasattr(self.risk_level, 'value') else self.risk_level
        return f"<ScanResult pkg={self.package_id} score={self.risk_score:.1f} [{level_val}]>"


class DetectorSignal(Base):
    __tablename__ = "detector_signals"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    scan_result_id = Column(Integer, ForeignKey("scan_results.id"), nullable=False)

    detector_name = Column(String(64), nullable=False)
    signal_value  = Column(Float, nullable=False)
    weight        = Column(Float, nullable=False)
    contribution  = Column(Float, nullable=False)
    detail        = Column(Text)

    scan_result = relationship("ScanResult", back_populates="signals")

    def __repr__(self):
        return f"<Signal {self.detector_name}={self.signal_value:.2f} w={self.weight}>"


class Allowlist(Base):
    __tablename__ = "allowlist"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    registry = Column(Enum(Registry, native_enum=False), nullable=False)
    name     = Column(String(256), nullable=False)
    added_by = Column(String(128), default="user")
    reason   = Column(Text)
    added_at = Column(DateTime, default=_utc_now)

    def __repr__(self):
        reg_val = self.registry.value if hasattr(self.registry, 'value') else self.registry
        return f"<Allowlist {reg_val}/{self.name}>"


# ── DB helpers ─────────────────────────────────────────────────────────────

def get_engine(db_url: str = "sqlite:///./supply_chain.db"):
    return create_engine(db_url, echo=False)

def init_db(db_url: str = "sqlite:///./supply_chain.db"):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine

def get_session(engine) -> Session:
    return Session(engine)