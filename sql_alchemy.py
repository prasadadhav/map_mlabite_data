import enum
from typing import List, Optional
from sqlalchemy import (
    create_engine, Column, ForeignKey, Table, Text, Boolean, String, Date, 
    Time, DateTime, Float, Integer, Enum
)
from sqlalchemy.ext.declarative import AbstractConcreteBase
from sqlalchemy.orm import (
    column_property, DeclarativeBase, Mapped, mapped_column, relationship
)
from datetime import datetime as dt_datetime, time as dt_time, date as dt_date

class Base(DeclarativeBase):
    pass

# Definitions of Enumerations
class UserRole(enum.Enum):
    tester = "tester"
    viewer = "viewer"
    admin = "admin"

class LicensingType(enum.Enum):
    Open_Source = "Open_Source"
    Proprietary = "Proprietary"

class ProjectStatus(enum.Enum):
    Closed = "Closed"
    Created = "Created"
    Archived = "Archived"
    Ready = "Ready"
    Pending = "Pending"

class EvaluationStatus(enum.Enum):
    Done = "Done"
    Archived = "Archived"
    Processing = "Processing"
    Custom = "Custom"
    Pending = "Pending"

class DatasetType(enum.Enum):
    Test = "Test"
    Validation = "Validation"
    Training = "Training"


# Tables definition for many-to-many relationships
evaluates_eval = Table(
    "evaluates_eval",
    Base.metadata,
    Column("evaluates", ForeignKey("element.id"), primary_key=True),
    Column("evalu", ForeignKey("evaluation.id"), primary_key=True),
)
model_dataset = Table(
    "model_dataset",
    Base.metadata,
    Column("models", ForeignKey("model.id"), primary_key=True),
    Column("dataset", ForeignKey("dataset.id"), primary_key=True),
)
derived_metric = Table(
    "derived_metric",
    Base.metadata,
    Column("derivedBy", ForeignKey("derived.id"), primary_key=True),
    Column("baseMetric", ForeignKey("metric.id"), primary_key=True),
)
evaluation_element = Table(
    "evaluation_element",
    Base.metadata,
    Column("eval", ForeignKey("evaluation.id"), primary_key=True),
    Column("ref", ForeignKey("element.id"), primary_key=True),
)
metriccategory_metric = Table(
    "metriccategory_metric",
    Base.metadata,
    Column("category", ForeignKey("metriccategory.id"), primary_key=True),
    Column("metrics", ForeignKey("metric.id"), primary_key=True),
)

# Tables definition
class test_catalog(Base):
    __tablename__ = "test_catalog"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    brief_description: Mapped[str] = mapped_column(String(100))
    dimension: Mapped[str] = mapped_column(String(100))

class landing_kpi(Base):
    __tablename__ = "landing_kpi"
    id: Mapped[int] = mapped_column(primary_key=True)
    partners: Mapped[int] = mapped_column(Integer)
    total_investment: Mapped[int] = mapped_column(Integer)
    focus_areas: Mapped[int] = mapped_column(Integer)
    entities: Mapped[int] = mapped_column(Integer)


class User(Base):
    __tablename__ = "user"
    role: Mapped[UserRole] = mapped_column(Enum(UserRole))
    first_name: Mapped[str] = mapped_column(String(100))
    last_name: Mapped[str] = mapped_column(String(100))
    password_hash: Mapped[str] = mapped_column(String(100))
    last_login: Mapped[dt_datetime] = mapped_column(DateTime)
    email: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[dt_datetime] = mapped_column(DateTime)
    id: Mapped[str] = mapped_column(String(100), primary_key=True)

class Comments(Base):
    __tablename__ = "comments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    comment: Mapped[str] = mapped_column(String(100))
    timeStamp: Mapped[dt_datetime] = mapped_column(DateTime)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))

class Evaluation(Base):
    __tablename__ = "evaluation"
    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[EvaluationStatus] = mapped_column(Enum(EvaluationStatus))
    config_id: Mapped[int] = mapped_column(ForeignKey("configuration.id"))
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"))

class Measure(Base):
    __tablename__ = "measure"
    id: Mapped[int] = mapped_column(primary_key=True)
    value: Mapped[str] = mapped_column(String(100))
    error: Mapped[str] = mapped_column(String(100))
    uncertainty: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(100))
    observation_id: Mapped[int] = mapped_column(ForeignKey("observation.id"))
    metric_id: Mapped[int] = mapped_column(ForeignKey("metric.id"))
    measurand_id: Mapped[int] = mapped_column(ForeignKey("element.id"))

class LegalRequirement(Base):
    __tablename__ = "legalrequirement"
    id: Mapped[int] = mapped_column(primary_key=True)
    legal_ref: Mapped[str] = mapped_column(String(100))
    standard: Mapped[str] = mapped_column(String(100))
    principle: Mapped[str] = mapped_column(String(100))
    project_1_id: Mapped[int] = mapped_column(ForeignKey("project.id"))

class AssessmentElement(AbstractConcreteBase, Base):
    strict_attrs = True
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))

class Element(AssessmentElement):
    __tablename__ = "element"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    project_id: Mapped[int] = mapped_column(ForeignKey("project.id"), nullable=True)
    type_spec: Mapped[str] = mapped_column(String(50))
    __mapper_args__ = {
        "polymorphic_identity": "element",
        "polymorphic_on": "type_spec",
    }

class Observation(AssessmentElement):
    __tablename__ = "observation"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    observer: Mapped[str] = mapped_column(String(100))
    whenObserved: Mapped[dt_datetime] = mapped_column(DateTime)
    eval_id: Mapped[int] = mapped_column(ForeignKey("evaluation.id"))
    tool_id: Mapped[int] = mapped_column(ForeignKey("tool.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id"))
    __mapper_args__ = {
        "polymorphic_identity": "observation",
        "concrete": True,
    }

class Tool(Base):
    __tablename__ = "tool"
    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(100))
    licensing: Mapped[LicensingType] = mapped_column(Enum(LicensingType))

class Metric(AssessmentElement):
    __tablename__ = "metric"
    id: Mapped[int] = mapped_column(primary_key=True)
    type_spec: Mapped[str] = mapped_column(String(50))
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    __mapper_args__ = {
        "polymorphic_identity": "metric",
        "polymorphic_on": "type_spec",
    }

class Direct(Metric):
    __tablename__ = "direct"
    id: Mapped[int] = mapped_column(ForeignKey("metric.id"), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    __mapper_args__ = {
        "polymorphic_identity": "direct",
    }

class ConfParam(AssessmentElement):
    __tablename__ = "confparam"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    param_type: Mapped[str] = mapped_column(String(100))
    value: Mapped[str] = mapped_column(String(100))
    conf_id: Mapped[int] = mapped_column(ForeignKey("configuration.id"))
    __mapper_args__ = {
        "polymorphic_identity": "confparam",
        "concrete": True,
    }

class MetricCategory(AssessmentElement):
    __tablename__ = "metriccategory"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    __mapper_args__ = {
        "polymorphic_identity": "metriccategory",
        "concrete": True,
    }

class Configuration(AssessmentElement):
    __tablename__ = "configuration"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    __mapper_args__ = {
        "polymorphic_identity": "configuration",
        "concrete": True,
    }

class Feature(Element):
    __tablename__ = "feature"
    id: Mapped[int] = mapped_column(ForeignKey("element.id"), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    feature_type: Mapped[str] = mapped_column(String(100))
    min_value: Mapped[float] = mapped_column(Float)
    max_value: Mapped[float] = mapped_column(Float)
    features_id: Mapped[int] = mapped_column(ForeignKey("datashape.id"))
    date_id: Mapped[int] = mapped_column(ForeignKey("datashape.id"))
    __mapper_args__ = {
        "polymorphic_identity": "feature",
    }

class Datashape(Base):
    __tablename__ = "datashape"
    id: Mapped[int] = mapped_column(primary_key=True)
    accepted_target_values: Mapped[str] = mapped_column(String(100))

class Dataset(Element):
    __tablename__ = "dataset"
    id: Mapped[int] = mapped_column(ForeignKey("element.id"), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(10000))
    source: Mapped[str] = mapped_column(String(100))
    version: Mapped[str] = mapped_column(String(100))
    licensing: Mapped[LicensingType] = mapped_column(Enum(LicensingType))
    dataset_type: Mapped[DatasetType] = mapped_column(Enum(DatasetType))
    datashape_id: Mapped[int] = mapped_column(ForeignKey("datashape.id"))
    __mapper_args__ = {
        "polymorphic_identity": "dataset",
    }

class Project(Base):
    __tablename__ = "project"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    status: Mapped[ProjectStatus] = mapped_column(Enum(ProjectStatus))

class Model(Element):
    __tablename__ = "model"
    id: Mapped[int] = mapped_column(ForeignKey("element.id"), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    pid: Mapped[str] = mapped_column(String(100))
    data: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(100))
    licensing: Mapped[LicensingType] = mapped_column(Enum(LicensingType))
    __mapper_args__ = {
        "polymorphic_identity": "model",
    }

class Derived(Metric):
    __tablename__ = "derived"
    id: Mapped[int] = mapped_column(ForeignKey("metric.id"), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(String(100))
    expression: Mapped[str] = mapped_column(String(100))
    __mapper_args__ = {
        "polymorphic_identity": "derived",
    }


#--- Relationships of the evaluation table
Evaluation.config: Mapped["Configuration"] = relationship("Configuration", back_populates="eval", foreign_keys=[Evaluation.config_id])
Evaluation.project: Mapped["Project"] = relationship("Project", back_populates="eval", foreign_keys=[Evaluation.project_id])
Evaluation.evaluates: Mapped[List["Element"]] = relationship("Element", secondary=evaluates_eval, back_populates="evalu")
Evaluation.observations: Mapped[List["Observation"]] = relationship("Observation", back_populates="eval", foreign_keys=[Observation.eval_id])
Evaluation.ref: Mapped[List["Element"]] = relationship("Element", secondary=evaluation_element, back_populates="eval")

#--- Relationships of the measure table
Measure.observation: Mapped["Observation"] = relationship("Observation", back_populates="measures", foreign_keys=[Measure.observation_id])
Measure.metric: Mapped["Metric"] = relationship("Metric", back_populates="measures", foreign_keys=[Measure.metric_id])
Measure.measurand: Mapped["Element"] = relationship("Element", back_populates="measure", foreign_keys=[Measure.measurand_id])

#--- Relationships of the comments table
Comments.user: Mapped["User"] = relationship("User", back_populates="comments", foreign_keys=[Comments.user_id])

#--- Relationships of the legalrequirement table
LegalRequirement.project_1: Mapped["Project"] = relationship("Project", back_populates="legal_requirements", foreign_keys=[LegalRequirement.project_1_id])

#--- Relationships of the element table
Element.project: Mapped["Project"] = relationship("Project", back_populates="involves", foreign_keys=[Element.project_id])
Element.measure: Mapped[List["Measure"]] = relationship("Measure", back_populates="measurand", foreign_keys=[Measure.measurand_id])
Element.eval: Mapped[List["Evaluation"]] = relationship("Evaluation", secondary=evaluation_element, back_populates="ref")
Element.evalu: Mapped[List["Evaluation"]] = relationship("Evaluation", secondary=evaluates_eval, back_populates="evaluates")

#--- Relationships of the observation table
Observation.measures: Mapped[List["Measure"]] = relationship("Measure", back_populates="observation", foreign_keys=[Measure.observation_id])
Observation.eval: Mapped["Evaluation"] = relationship("Evaluation", back_populates="observations", foreign_keys=[Observation.eval_id])
Observation.tool: Mapped["Tool"] = relationship("Tool", back_populates="observation_1", foreign_keys=[Observation.tool_id])
Observation.dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="observation_2", foreign_keys=[Observation.dataset_id])

#--- Relationships of the tool table
Tool.observation_1: Mapped[List["Observation"]] = relationship("Observation", back_populates="tool", foreign_keys=[Observation.tool_id])

#--- Relationships of the metric table
Metric.measures: Mapped[List["Measure"]] = relationship("Measure", back_populates="metric", foreign_keys=[Measure.metric_id])
Metric.category: Mapped[List["MetricCategory"]] = relationship("MetricCategory", secondary=metriccategory_metric, back_populates="metrics")
Metric.derivedBy: Mapped[List["Derived"]] = relationship("Derived", secondary=derived_metric, back_populates="baseMetric")

#--- Relationships of the confparam table
ConfParam.conf: Mapped["Configuration"] = relationship("Configuration", back_populates="params", foreign_keys=[ConfParam.conf_id])

#--- Relationships of the metriccategory table
MetricCategory.metrics: Mapped[List["Metric"]] = relationship("Metric", secondary=metriccategory_metric, back_populates="category")

#--- Relationships of the configuration table
Configuration.eval: Mapped[List["Evaluation"]] = relationship("Evaluation", back_populates="config", foreign_keys=[Evaluation.config_id])
Configuration.params: Mapped[List["ConfParam"]] = relationship("ConfParam", back_populates="conf", foreign_keys=[ConfParam.conf_id])

#--- Relationships of the feature table
Feature.features: Mapped["Datashape"] = relationship("Datashape", back_populates="f_features", foreign_keys=[Feature.features_id])
Feature.date: Mapped["Datashape"] = relationship("Datashape", back_populates="f_date", foreign_keys=[Feature.date_id])

#--- Relationships of the datashape table
Datashape.dataset_1: Mapped[List["Dataset"]] = relationship("Dataset", back_populates="datashape", foreign_keys=[Dataset.datashape_id])
Datashape.f_features: Mapped[List["Feature"]] = relationship("Feature", back_populates="features", foreign_keys=[Feature.features_id])
Datashape.f_date: Mapped[List["Feature"]] = relationship("Feature", back_populates="date", foreign_keys=[Feature.date_id])

#--- Relationships of the dataset table
Dataset.models: Mapped[List["Model"]] = relationship("Model", secondary=model_dataset, back_populates="dataset")
Dataset.datashape: Mapped["Datashape"] = relationship("Datashape", back_populates="dataset_1", foreign_keys=[Dataset.datashape_id])
Dataset.observation_2: Mapped[List["Observation"]] = relationship("Observation", back_populates="dataset", foreign_keys=[Observation.dataset_id])

#--- Relationships of the project table
Project.involves: Mapped[List["Element"]] = relationship("Element", back_populates="project", foreign_keys=[Element.project_id])
Project.eval: Mapped[List["Evaluation"]] = relationship("Evaluation", back_populates="project", foreign_keys=[Evaluation.project_id])
Project.legal_requirements: Mapped[List["LegalRequirement"]] = relationship("LegalRequirement", back_populates="project_1", foreign_keys=[LegalRequirement.project_1_id])

#--- Relationships of the model table
Model.dataset: Mapped[List["Dataset"]] = relationship("Dataset", secondary=model_dataset, back_populates="models")

#--- Relationships of the derived table
Derived.baseMetric: Mapped[List["Metric"]] = relationship("Metric", secondary=derived_metric, back_populates="derivedBy")

# Database connection
DATABASE_URL = "sqlite:///MLABite_Mar_2026.db"  # SQLite connection
engine = create_engine(DATABASE_URL, echo=True)

# Create tables in the database
Base.metadata.create_all(engine, checkfirst=True)