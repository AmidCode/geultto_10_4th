from sqlalchemy import (
    Table,
    Column,
    Integer,
    String,
    ForeignKey,
    UUID,
    Enum,
)
from sqlalchemy.orm import (
    registry,
    relationship,
)

from third.io.query.domain import (
    User,
    UserInfo,
    UserProfile,
    Gender,
)

# Registry와 MetaData 객체 생성
mapper_registry = registry()
metadata = mapper_registry.metadata

# 테이블 정의
t_users = Table(
    "users_tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("uid", UUID, unique=True, nullable=False),
)

t_user_info = Table(
    "user_info_tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("uid", UUID, ForeignKey("users_tbl.uid")),
    Column("name", String(50), unique=True, nullable=False),
    Column("age", Integer, nullable=False),
    Column("gender", Enum(Gender), nullable=False),
)

t_user_profile = Table(
    "user_profile_tbl",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("uid", UUID, ForeignKey("users_tbl.uid")),
    Column("nickname", String(50), nullable=False),
    Column("pic", String(256), nullable=True),
)


mapper_registry.map_imperatively(
    User,
    t_users,
    properties={
        "info": relationship(
            UserInfo,
            uselist=False,
            cascade="all, delete-orphan",
        ),
        "profile": relationship(
            UserProfile,
            uselist=False,
            cascade="all, delete-orphan",
        ),
    },
)

mapper_registry.map_imperatively(
    UserInfo,
    t_user_info,
)

mapper_registry.map_imperatively(
    UserProfile,
    t_user_profile,
)
